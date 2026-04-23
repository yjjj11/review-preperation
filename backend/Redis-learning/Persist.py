import redis

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# ----------------------
# 1. 读取 RDB 配置
# ----------------------
print("=== RDB 配置 ===")
rdb_save = r.config_get("save")["save"]  # 获取 RDB 触发规则
rdb_compression = r.config_get("rdbcompression")["rdbcompression"]  # 是否压缩 RDB
print(f"RDB 触发规则: {rdb_save}")  # 输出如 "900 1 300 10 60 10000"（900秒1个key变化则保存）
print(f"RDB 压缩: {rdb_compression}")  # 输出 yes/no

# ----------------------
# 2. 读取 AOF 配置
# ----------------------
print("\n=== AOF 配置 ===")
aof_enabled = r.config_get("appendonly")["appendonly"]  # 是否开启 AOF
aof_fsync = r.config_get("appendfsync")["appendfsync"]  # AOF 刷盘策略
print(f"AOF 开启状态: {aof_enabled}")  # 输出 yes/no
print(f"AOF 刷盘策略: {aof_fsync}")  # 输出 always/everysec/no（everysec 是推荐值）

# ----------------------
# 3. 修改配置（运行时生效，重启失效）
# ----------------------
print("\n=== 修改配置 ===")
# 修改 RDB 触发规则：60秒内至少1个key变化则保存
r.config_set("save", "60 1")
# 开启 AOF
r.config_set("appendonly", "yes")
# 修改 AOF 刷盘策略为 everysec（每秒刷盘，性能与安全平衡）
r.config_set("appendfsync", "everysec")

# 验证修改
print("修改后的 RDB 触发规则:", r.config_get("save")["save"])
print("修改后的 AOF 开启状态:", r.config_get("appendonly")["appendonly"])

# ----------------------
# 4. 将配置永久保存到 redis.conf（需 Redis 有写权限）
# ----------------------
# r.config_rewrite()  # 取消注释这行，将运行时配置写入 redis.conf
print("\n配置已修改（运行时生效），如需永久保存请执行 config_rewrite()")