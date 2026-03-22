import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from networks import Actor, Critic
from replay_buffer import ReplayBuffer


class MADDPG:
    """
    Multi-Agent DDPG cho SISO RSMA
    2 agents: Agent1 (BS1) và Agent2 (BS2)
    """
    def __init__(self,
                 state_dim   = 4,
                 action_dim  = 2,
                 P_total     = 1.0,
                 hidden_dim  = 64,
                 lr_actor    = 5e-5,
                 lr_critic   = 5e-5,
                 gamma       = 0.99,
                 tau         = 0.01,
                 buffer_size = 15000,
                 batch_size  = 128):

        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size

        # ── Chọn device (GPU nếu có, không thì CPU) ──
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Dùng device: {self.device}")

        # ── Tạo Actor networks (2 agents) ─────────────
        self.actor1  = Actor(state_dim, action_dim,
                             hidden_dim, P_total).to(self.device)
        self.actor2  = Actor(state_dim, action_dim,
                             hidden_dim, P_total).to(self.device)

        # Target networks: copy từ actor, update chậm
        self.actor1_target = Actor(state_dim, action_dim,
                                   hidden_dim, P_total).to(self.device)
        self.actor2_target = Actor(state_dim, action_dim,
                                   hidden_dim, P_total).to(self.device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())
        self.actor2_target.load_state_dict(self.actor2.state_dict())

        # ── Tạo Critic networks (2 agents) ────────────
        self.critic1 = Critic(state_dim, action_dim,
                              hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim,
                              hidden_dim).to(self.device)

        self.critic1_target = Critic(state_dim, action_dim,
                                     hidden_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim,
                                     hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # ── Optimizers ────────────────────────────────
        self.opt_actor1  = optim.Adam(self.actor1.parameters(),
                                      lr=lr_actor)
        self.opt_actor2  = optim.Adam(self.actor2.parameters(),
                                      lr=lr_actor)
        self.opt_critic1 = optim.Adam(self.critic1.parameters(),
                                      lr=lr_critic)
        self.opt_critic2 = optim.Adam(self.critic2.parameters(),
                                      lr=lr_critic)

        # ── Replay Buffer ─────────────────────────────
        self.buffer = ReplayBuffer(buffer_size,
                                   state_dim, action_dim)

    # ══════════════════════════════════════════════
    # CHỌN ACTION (dùng khi tương tác với env)
    # ══════════════════════════════════════════════
    def select_action(self, s1, s2, noise_std=0.1):
        """
        Chọn action từ Actor + thêm noise để exploration
        noise_std: độ lớn noise (giảm dần theo training)
        """
        s1t = torch.FloatTensor(s1).unsqueeze(0).to(self.device)
        s2t = torch.FloatTensor(s2).unsqueeze(0).to(self.device)

        self.actor1.eval()
        self.actor2.eval()
        with torch.no_grad():
            a1 = self.actor1(s1t).cpu().numpy()[0]
            a2 = self.actor2(s2t).cpu().numpy()[0]
        self.actor1.train()
        self.actor2.train()

        # Thêm noise để exploration
        a1 += np.random.normal(0, noise_std, size=a1.shape)
        a2 += np.random.normal(0, noise_std, size=a2.shape)

        # Clip để action không âm, rồi normalize tổng = P_total
        a1 = self._normalize_action(a1)
        a2 = self._normalize_action(a2)
        return a1, a2

    def _normalize_action(self, action, P_total=1.0):
        """Đảm bảo action ≥ 0 và tổng = P_total"""
        action = np.clip(action, 1e-6, None)   # không âm
        action = action / action.sum() * P_total  # normalize
        return action.astype(np.float32)

    # ══════════════════════════════════════════════
    # LEARN (cập nhật networks từ replay buffer)
    # ══════════════════════════════════════════════
    def learn(self):
        """Được gọi sau mỗi step, khi buffer đã đủ"""
        if not self.buffer.ready(self.batch_size):
            return None, None   # chưa đủ data

        # ── ① Sample batch ────────────────────────
        s1, s2, a1, a2, r, s1n, s2n = self.buffer.sample(
            self.batch_size
        )

        # Chuyển sang tensor
        s1  = torch.FloatTensor(s1).to(self.device)
        s2  = torch.FloatTensor(s2).to(self.device)
        a1  = torch.FloatTensor(a1).to(self.device)
        a2  = torch.FloatTensor(a2).to(self.device)
        r   = torch.FloatTensor(r).to(self.device)
        s1n = torch.FloatTensor(s1n).to(self.device)
        s2n = torch.FloatTensor(s2n).to(self.device)

        # ── ② Cập nhật Critic ─────────────────────
        with torch.no_grad():
            # Action tiếp theo từ target actor
            a1_next = self.actor1_target(s1n)
            a2_next = self.actor2_target(s2n)

            # Target Q từ target critic
            q1_next = self.critic1_target(s1n, s2n,
                                          a1_next, a2_next)
            q2_next = self.critic2_target(s1n, s2n,
                                          a1_next, a2_next)

            # Bellman equation: y = r + γ * Q_target(s', a')
            y = r + self.gamma * q1_next  # dùng chung reward

        # Q hiện tại
        q1_curr = self.critic1(s1, s2, a1, a2)
        q2_curr = self.critic2(s1, s2, a1, a2)

        # Loss = MSE(Q_hiện_tại, y)
        loss_c1 = F.mse_loss(q1_curr, y)
        loss_c2 = F.mse_loss(q2_curr, y)

        self.opt_critic1.zero_grad()
        loss_c1.backward()
        self.opt_critic1.step()

        self.opt_critic2.zero_grad()
        loss_c2.backward()
        self.opt_critic2.step()

        # ── ③ Cập nhật Actor ──────────────────────
        # Actor loss = -Q (muốn maximize Q)
        a1_pred = self.actor1(s1)
        a2_pred = self.actor2(s2)

        loss_a1 = -self.critic1(s1, s2, a1_pred, a2.detach()).mean()
        loss_a2 = -self.critic2(s1, s2, a1.detach(), a2_pred).mean()

        self.opt_actor1.zero_grad()
        loss_a1.backward()
        self.opt_actor1.step()

        self.opt_actor2.zero_grad()
        loss_a2.backward()
        self.opt_actor2.step()

        # ── ④ Soft update target networks ─────────
        self._soft_update(self.actor1,  self.actor1_target)
        self._soft_update(self.actor2,  self.actor2_target)
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return loss_c1.item(), loss_a1.item()

    def _soft_update(self, net, net_target):
        """θ_target ← τ*θ + (1-τ)*θ_target"""
        for p, p_t in zip(net.parameters(),
                          net_target.parameters()):
            p_t.data.copy_(
                self.tau * p.data + (1 - self.tau) * p_t.data
            )

    # ══════════════════════════════════════════════
    # LƯU / NẠP MODEL
    # ══════════════════════════════════════════════
    def save(self, path="model"):
        torch.save(self.actor1.state_dict(),  f"{path}_a1.pth")
        torch.save(self.actor2.state_dict(),  f"{path}_a2.pth")
        print(f"Đã lưu model vào {path}_a1.pth, {path}_a2.pth")

    def load(self, path="model"):
        self.actor1.load_state_dict(
            torch.load(f"{path}_a1.pth",
                       map_location=self.device))
        self.actor2.load_state_dict(
            torch.load(f"{path}_a2.pth",
                       map_location=self.device))
        print(f"Đã nạp model từ {path}")