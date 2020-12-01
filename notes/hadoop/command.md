### 系统管理

```bash
# 启动/停止 HDFS
start-dfs.sh
stop-dfs.sh

# 启动/关闭 YARN
start-yarn.sh
stop-yarn.sh

# 进入/退出 安全模式
hdfs dfsadmin -safemode enter
hdfs dfsadmin -safemode leave
```

### 文件管理

```bash
# 创建目录
hdfs dfs -mkdir <path>

# 上传到HDFS
hdfs dfs -put <localsrc> <dst>

# 复制到HDFS
hdfs dfs -copyFromLocal <localsrc> <dst>

# 查看目录
hdfs dfs -ls <path>

# 查看文件
hdfs dfs -cat <src>

# 下载文件
hdfs dfs -get  <src> <localdst>

# 合并文件
hdfs dfs -getmerge <src> <localdst>

# 删除文件
hdfs dfs -rm -r <path>
```