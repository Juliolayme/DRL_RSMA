import numpy as np

class SISO_RSMA_Env:
    """
    SISO 2-user Interference Channel với RSMA
    
    Topology:
      BS1 --h1--> UE1  (kênh muốn)
      BS2 --g2--> UE1  (nhiễu)
      BS2 --h2--> UE2  (kênh muốn)
      BS1 --g1--> UE2  (nhiễu)
    """
    def __init__(self, P_total=1.0, noise_power=0.1, beta=0.5):
        self.P_total   = P_total
        self.N0        = noise_power
        self.beta      = beta
        self.state_dim  = 4   # [Re(h), Im(h), Re(g), Im(g)]
        self.action_dim = 2   # [P_common, P_private]
        
        # Kênh truyền
        self.h1 = self.h2 = self.g1 = self.g2 = None

    # ────────────────────────────────────────────
    # RESET: sinh kênh Rayleigh mới mỗi episode
    # ────────────────────────────────────────────
    def reset(self):
        self.h1 = self._gen_channel()
        self.h2 = self._gen_channel()
        self.g1 = self._gen_channel()
        self.g2 = self._gen_channel()
        return self._get_states()

    def _gen_channel(self):
        """Rayleigh fading: CN(0,1)"""
        return (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    def _get_states(self):
        s1 = np.array([self.h1.real, self.h1.imag,
                       self.g1.real, self.g1.imag], dtype=np.float32)
        s2 = np.array([self.h2.real, self.h2.imag,
                       self.g2.real, self.g2.imag], dtype=np.float32)
        return s1, s2

    # ────────────────────────────────────────────
    # STEP: nhận action, trả về reward
    # ────────────────────────────────────────────
    def step(self, action1, action2):
        P1c, P1p = float(action1[0]), float(action1[1])
        P2c, P2p = float(action2[0]), float(action2[1])

        # Tính reward với kênh hiện tại
        best_reward = -np.inf
        best_R1 = best_R2 = 0.0
        for o1 in [0, 1]:
            for o2 in [0, 1]:
                R1, R2 = self._compute_rate(
                    P1c, P1p, P2c, P2p, o1, o2)
                r = self.beta*R1 + (1-self.beta)*R2
                if r > best_reward:
                    best_reward, best_R1, best_R2 = r, R1, R2

        # FIX: sinh kênh MỚI sau mỗi step
        # → next_state khác state → agent học được
        self.h1 = self._gen_channel()
        self.h2 = self._gen_channel()
        self.g1 = self._gen_channel()
        self.g2 = self._gen_channel()

        s1_next, s2_next = self._get_states()
        return s1_next, s2_next, best_reward, best_R1, best_R2

    # ────────────────────────────────────────────
    # COMPUTE RATE: công thức Shannon với RSMA
    # ────────────────────────────────────────────
    def _compute_rate(self, P1c, P1p, P2c, P2p, order1, order2):
        h1 = abs(self.h1)**2   # |h1|²
        h2 = abs(self.h2)**2
        g1 = abs(self.g1)**2
        g2 = abs(self.g2)**2
        N0 = self.N0

        # ── Rate tại UE1 ──────────────────────
        if order1 == 0:   # b2c → b1c → b1p
            R1_2c = np.log2(1 + g2*P2c /
                            (h1*(P1c+P1p) + g2*P2p + N0))
            R1_1c = np.log2(1 + h1*P1c /
                            (h1*P1p + g2*P2p + N0))
        else:             # b1c → b2c → b1p
            R1_1c = np.log2(1 + h1*P1c /
                            (h1*P1p + g2*(P2c+P2p) + N0))
            R1_2c = np.log2(1 + g2*P2c /
                            (h1*P1p + g2*P2p + N0))

        R1p = np.log2(1 + h1*P1p / (g2*P2p + N0))

        # ── Rate tại UE2 ──────────────────────
        if order2 == 0:   # b1c → b2c → b2p
            R2_1c = np.log2(1 + g1*P1c /
                            (g1*P1p + h2*(P2c+P2p) + N0))
            R2_2c = np.log2(1 + h2*P2c /
                            (g1*P1p + h2*P2p + N0))
        else:             # b2c → b1c → b2p
            R2_2c = np.log2(1 + h2*P2c /
                            (g1*(P1c+P1p) + h2*P2p + N0))
            R2_1c = np.log2(1 + g1*P1c /
                            (g1*P1p + h2*P2p + N0))

        R2p = np.log2(1 + h2*P2p / (g1*P1p + N0))

        # ── Common rate = min (phải decode được ở cả 2 UE) ──
        R1c = min(R1_1c, R2_1c)
        R2c = min(R1_2c, R2_2c)

        R1 = R1c + R1p
        R2 = R2c + R2p
        return R1, R2