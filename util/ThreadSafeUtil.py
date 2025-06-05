import threading


class ThreadSafeCounter:
    """线程安全的计数器"""
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def add(self, amount):
        """原子性地增加计数"""
        with self._lock:
            self._value += amount
            return self._value
    
    def subtract(self, amount):
        """原子性地减少计数"""
        with self._lock:
            self._value -= amount
            return self._value
    
    def set(self, value):
        """原子性地设置值"""
        with self._lock:
            self._value = value
            return self._value
    
    def reset(self):
        """原子性地重置为0"""
        with self._lock:
            self._value = 0
            return self._value
    
    @property
    def value(self):
        """获取当前值"""
        with self._lock:
            return self._value


class ThreadSafeDict:
    """线程安全的字典"""
    def __init__(self):
        self._dict = {}
        self._lock = threading.Lock()
    
    def get(self, key, default=None):
        """获取值"""
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key, value):
        """设置值"""
        with self._lock:
            self._dict[key] = value
    
    def update(self, key, value):
        """更新值（同set）"""
        with self._lock:
            self._dict[key] = value
    
    def delete(self, key):
        """删除键值对"""
        with self._lock:
            if key in self._dict:
                del self._dict[key]
    
    def keys(self):
        """获取所有键"""
        with self._lock:
            return list(self._dict.keys())
    
    def values(self):
        """获取所有值"""
        with self._lock:
            return list(self._dict.values())
    
    def items(self):
        """获取所有键值对"""
        with self._lock:
            return list(self._dict.items())
    
    def __len__(self):
        """获取长度"""
        with self._lock:
            return len(self._dict) 