import numpy as np


class ReplayBuffer:
    """
    Lưu trữ experience của CẢ 2 agent
    
    Mỗi experience = (s1, s2, a1, a2, reward, s1', s2')
    """
    def __init__(self, max_size=15000,
                 state_dim=4, action_dim=2):
        self.max_size   = max_size
        self.ptr        = 0      # con trỏ vị trí hiện tại
        self.size       = 0      # số experience đang lưu

        # Pre-allocate memory (nhanh hơn append)
        self.s1      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.s2      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.a1      = np.zeros((max_size, action_dim), dtype=np.float32)
        self.a2      = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward  = np.zeros((max_size, 1),          dtype=np.float32)
        self.s1_next = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.s2_next = np.zeros((max_size, state_dim),  dtype=np.float32)

    def push(self, s1, s2, a1, a2, reward, s1_next, s2_next):
        """Lưu 1 experience vào buffer"""
        self.s1[self.ptr]      = s1
        self.s2[self.ptr]      = s2
        self.a1[self.ptr]      = a1
        self.a2[self.ptr]      = a2
        self.reward[self.ptr]  = reward
        self.s1_next[self.ptr] = s1_next
        self.s2_next[self.ptr] = s2_next

        # Vòng tròn: khi đầy thì ghi đè experience cũ nhất
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=128):
        """Lấy ngẫu nhiên batch_size experience"""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.s1[idx],
            self.s2[idx],
            self.a1[idx],
            self.a2[idx],
            self.reward[idx],
            self.s1_next[idx],
            self.s2_next[idx]
        )

    def ready(self, batch_size=128):
        """Kiểm tra đã đủ experience để train chưa"""
        return self.size >= batch_size

    def __len__(self):
        return self.size