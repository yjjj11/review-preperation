import redis
import time
import uuid

# 连接Redis（单机/集群通用，集群模式下key加Hash Tag，如{rate_limit}:user:xxx）
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def is_sliding_window_limited(
    user_id: int,        # 限流主体：用户ID/IP/接口名
    api_path: str,       # 限流的接口
    limit: int = 10,     # 窗口内最大请求数
    window_seconds: int = 60  # 窗口大小（秒），这里是1分钟
) -> bool:
    """
    滑动窗口限流核心函数
    返回True：触发限流，拒绝请求；返回False：放行请求
    """
    # 1. 定义Redis key，按用户+接口隔离限流规则
    # 集群模式下建议写成：key = f"{{rate_limit}}:{user_id}:{api_path}"
    key = f"rate_limit:{user_id}:{api_path}"
    
    # 2. 计算时间边界
    current_time_ms = time.time() * 1000  # 当前时间戳（毫秒，保证精度）
    window_start_ms = current_time_ms - (window_seconds * 1000)  # 窗口左边界（早于这个时间的请求都要删掉）

    # 3. 用Pipeline打包所有操作，保证原子性！（关键！防止并发下统计错误）
    pipe = r.pipeline(transaction=True)
    
    # 步骤1：删除窗口外的所有旧请求（只保留当前窗口内的记录）
    pipe.zremrangebyscore(key, 0, window_start_ms)
    
    # 步骤2：统计当前窗口内的总请求数
    pipe.zcard(key)
    
    # 步骤3：把当前请求记录到Sorted Set里
    # member用uuid保证唯一，防止同一毫秒的多个请求被覆盖；score用当前时间戳
    pipe.zadd(key, {str(uuid.uuid4()): current_time_ms})
    
    # 步骤4：给key设置过期时间（窗口大小+1秒，防止冷数据占用内存）
    pipe.expire(key, window_seconds + 1)
    
    # 4. 执行所有命令，按顺序获取结果
    _, current_request_count, _, _ = pipe.execute()

    # 5. 判断是否超过限流阈值
    if current_request_count >= limit:
        print(f"触发限流！当前窗口内请求数：{current_request_count}，限制：{limit}")
        return True
    else:
        print(f"请求放行！当前窗口内请求数：{current_request_count}，限制：{limit}")
        return False

# ----------------------
# 测试代码
# ----------------------
if __name__ == "__main__":
    # 模拟同一个用户1分钟内访问15次接口，限制10次
    for i in range(15):
        is_limited = is_sliding_window_limited(
            user_id=1001,
            api_path="/api/v1/order",
            limit=10,
            window_seconds=60
        )
        # 模拟请求间隔
        time.sleep(0.1)