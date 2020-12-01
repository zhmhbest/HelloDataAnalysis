
### 单机部署

仅1个节点运行1个java进程，主要用于调试。

#### demo

```bash
workspace=/tmp/demo
if [ ! -e $workspace ]; then mkdir $workspace; fi
pushd $workspace

mkdir input
cp $HADOOP_HOME/etc/hadoop/*.xml input

# 例子
example_jar=$(echo $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar); echo $example_jar

# 单机模式运行MapReduce
hadoop jar $example_jar wordcount input 'dfs[az]+'

popd
rm -rf "$workspace"
```

### 伪分布部署

相关守护进程都独立运行，但运行于同一台计算机上

`core-site.xml`

```bash
vim $HADOOP_HOME/etc/hadoop/core-site.xml
```

```xml
<configuration>
    <!-- ↓↓↓追加内容↓↓↓ -->
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
    <!-- ↑↑↑追加内容↑↑↑ -->
</configuration>
```

`hdfs-site.xml`

```bash
vim $HADOOP_HOME/etc/hadoop/hdfs-site.xml
```

```xml
<configuration>
    <!-- ↓↓↓追加内容↓↓↓ -->
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <!-- ↑↑↑追加内容↑↑↑ -->
</configuration>
```

`hadoop-env.sh`

```bash
vim $HADOOP_HOME/etc/hadoop/hadoop-env.sh
```

```sh
# 守护进程
# - NameNode：管理文件系统名称空间和对集群中存储的文件的访问
# - SecondaryNamenode：提供周期检查点和清理任务，一般运行在一台非NameNode的机器上
# - DataNode：管理连接到节点的存储（一个集群中可以有多个节点）
# - ResourceManager：负责全局的资源管理和任务调度，把整个集群当成计算资源池，只关注分配，不管应用，且不负责容错
# - NodeManager：执行在单个节点上的代理，它管理Hadoop集群中单个计算节点

# ↓↓↓追加内容↓↓↓
HDFS_NAMENODE_USER=root
HDFS_SECONDARYNAMENODE_USER=root
HDFS_DATANODE_USER=root
YARN_RESOURCEMANAGER_USER=root
YARN_NODEMANAGER_USER=root
# ↑↑↑追加内容↑↑↑
```

`authorized_keys`

```bash
ssh-keygen -t rsa -P ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

#### 运行

```bash
# 格式化文件系统
hdfs namenode -format

# /************************************************************
# SHUTDOWN_MSG: Shutting down NameNode at localhost/127.0.0.1
# ************************************************************/

# 启动HDFS
start-dfs.sh

# 启动YARN
start-yarn.sh

# 查看已启动的Hadoop进程
jps

# 关闭防火墙
systemctl stop firewalld

# 查看访问IP
ifconfig | grep 'inet '
# 访问<ip>:9870
```
