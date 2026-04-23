import redis

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# ----------------------
# 1. 普通 Pipeline：仅批量执行，不保证原子性（性能最高）
# ----------------------
pipe = r.pipeline(transaction=False)  # transaction=False 关闭事务
pipe.set("pipe_key1", "value1")
pipe.set("pipe_key2", "value2")
pipe.get("pipe_key1")
pipe.get("pipe_key2")
results = pipe.execute()  # 一次性发送所有命令
print("普通 Pipeline 结果:", results)  # 输出 [True, True, 'value1', 'value2']

# ----------------------
# 2. 事务 Pipeline：保证原子性（multi/exec）
#不支持回滚，只能保证并发情况下服务端是处理完了该事务再去处理别的命令
#不会出现事务执行过程中被别的命令插队的情况
# ----------------------
pipe = r.pipeline(transaction=True)  
try:
    pipe.multi()  # 显式开启事务（可选，pipeline 默认会自动开启）
    pipe.set("tx_key5", "tx_value1")
    pipe.set("tx_key6", "tx_value2")
    results = pipe.execute()  # 执行事务
    print("事务执行成功:", results)  # 输出 [True, True]
except Exception as e:
    print("事务执行失败:", e)

# 验证事务结果
print("tx_key5:", r.get("tx_key5"))  
print("tx_key6:", r.get("tx_key6"))  