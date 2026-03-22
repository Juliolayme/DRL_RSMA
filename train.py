import numpy as np
import matplotlib.pyplot as plt
from environment import SISO_RSMA_Env
from maddpg import MADDPG


# ══════════════════════════════════════════════
# HYPERPARAMETERS (giống paper Table II)
# ══════════════════════════════════════════════
CONFIG = {
    "episodes"    : 12000,   # số episode training
    "steps"       : 200,     # số step mỗi episode
    "noise_start" : 0.3,     # noise ban đầu (exploration nhiều)
    "noise_end"   : 0.01,    # noise cuối (exploitation nhiều)
    "noise_decay" : 0.9998,  # tốc độ giảm noise
    "log_interval": 200,     # in kết quả mỗi N episode
    "save_interval": 2000,   # lưu model mỗi N episode
    "P_total"     : 1.0,
    "noise_power" : 0.1,     # N0 (SNR=10 tương đương)
    "beta"        : 0.5,
}


def train():
    # ── Khởi tạo ──────────────────────────────
    env   = SISO_RSMA_Env(
        P_total     = CONFIG["P_total"],
        noise_power = CONFIG["noise_power"],
        beta        = CONFIG["beta"]
    )
    agent = MADDPG(
        P_total    = CONFIG["P_total"],
        lr_actor   = 5e-5,
        lr_critic  = 5e-5,
        gamma      = 0.99,
        tau        = 0.01,
        batch_size = 128
    )

    # ── Lưu kết quả để vẽ đồ thị ─────────────
    reward_history = []   # reward trung bình mỗi episode
    R1_history     = []
    R2_history     = []

    noise_std = CONFIG["noise_start"]
    best_reward = -np.inf

    print("=" * 50)
    print("BẮT ĐẦU TRAINING MADDPG-RSMA")
    print(f"Episodes: {CONFIG['episodes']}, "
          f"Steps/ep: {CONFIG['steps']}")
    print("=" * 50)

    # ══════════════════════════════════════════
    # VÒNG LẶP TRAINING CHÍNH
    # ══════════════════════════════════════════
    for ep in range(1, CONFIG["episodes"] + 1):

        s1, s2 = env.reset()
        ep_reward = 0.0
        ep_R1     = 0.0
        ep_R2     = 0.0

        for step in range(CONFIG["steps"]):

            # ① Agent chọn action
            a1, a2 = agent.select_action(s1, s2,
                                         noise_std=noise_std)

            # ② Tương tác với environment
            s1n, s2n, reward, R1, R2 = env.step(a1, a2)

            # ③ Lưu experience vào buffer
            agent.buffer.push(s1, s2, a1, a2,
                               reward, s1n, s2n)

            # ④ Học từ buffer
            agent.learn()

            # Cập nhật state
            s1, s2     = s1n, s2n
            ep_reward += reward
            ep_R1     += R1
            ep_R2     += R2

        # ── Tính trung bình mỗi episode ───────
        avg_reward = ep_reward / CONFIG["steps"]
        avg_R1     = ep_R1     / CONFIG["steps"]
        avg_R2     = ep_R2     / CONFIG["steps"]

        reward_history.append(avg_reward)
        R1_history.append(avg_R1)
        R2_history.append(avg_R2)

        # ── Giảm noise theo thời gian ─────────
        noise_std = max(CONFIG["noise_end"],
                        noise_std * CONFIG["noise_decay"])

        # ── Lưu model tốt nhất ────────────────
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save("best_model")

        # ── In kết quả ────────────────────────
        if ep % CONFIG["log_interval"] == 0:
            recent_avg = np.mean(reward_history[-200:])
            print(f"Ep {ep:5d} | "
                f"Reward: {avg_reward:.3f} | "
                f"R1: {avg_R1:.3f} | "
                f"R2: {avg_R2:.3f} | "
                f"Balance: {min(avg_R1,avg_R2)/max(avg_R1,avg_R2):.2f} | "  # ← thêm dòng này
                f"Avg200: {recent_avg:.3f} | "
                f"Noise: {noise_std:.4f}")

        # ── Lưu model định kỳ ─────────────────
        if ep % CONFIG["save_interval"] == 0:
            agent.save(f"model_ep{ep}")

    # ══════════════════════════════════════════
    # VẼ ĐỒ THỊ KẾT QUẢ
    # ══════════════════════════════════════════
    plot_results(reward_history, R1_history, R2_history)
    print(f"\nBest reward đạt được: {best_reward:.4f}")
    return agent, reward_history


def plot_results(reward_history, R1_history, R2_history):
    """Vẽ learning curve giống Fig.5 trong paper"""

    # Smooth bằng moving average
    def moving_avg(data, window=200):
        return np.convolve(data,
                           np.ones(window)/window,
                           mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot 1: Reward ─────────────────────
    axes[0].plot(reward_history,
                 alpha=0.3, color='blue', label='Raw')
    if len(reward_history) >= 200:
        axes[0].plot(moving_avg(reward_history),
                     color='blue', linewidth=2,
                     label='Moving Avg (200)')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Weighted Sum-Rate (bits/s/Hz)")
    axes[0].set_title("Learning Curve — MADDPG RSMA")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Plot 2: R1 vs R2 ───────────────────
    axes[1].plot(R1_history,
                 alpha=0.3, color='red',   label='R1 raw')
    axes[1].plot(R2_history,
                 alpha=0.3, color='green', label='R2 raw')
    if len(R1_history) >= 200:
        axes[1].plot(moving_avg(R1_history),
                     color='red',   linewidth=2,
                     label='R1 avg')
        axes[1].plot(moving_avg(R2_history),
                     color='green', linewidth=2,
                     label='R2 avg')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rate (bits/s/Hz)")
    axes[1].set_title("Rate per User")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()
    print("Đã lưu đồ thị: learning_curve.png")


# ══════════════════════════════════════════════
# CHẠY TRAINING
# ══════════════════════════════════════════════
if __name__ == "__main__":
    agent, history = train()