<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Hadoop](../index.html)

[TOC]

## 概念

[Hadoop](http://hadoop.apache.org/)是一个由Apache基金会开发的框架，该框架允许使用简单的编程模型**跨计算机集群**对**大型数据集**进行**分布式处理**。其模块包括

- Hadoop Common
- HDFS（Hadoop Distributed File System）
- Hadoop YARN
- Hadoop MapReduce
- Hadoop Ozone

### Hadoop生态

- HDFS：作为文件系统存放在最底层
- YARN：资源调度器
- [HBase](http://hbase.apache.org/downloads.html)：分布式列存储数据库
- Hlive：数据仓库工具，可将SQL转换成MapReduce任务
- R Connectors：使用R语言访问
- Mahout：机器学习算法库
- Pig：数据分析工具
- Oozie：基于工作流引擎的服务器
- ZooKeeper：分布式应用程序协调服务
- Flume：分布式日志采集聚合传输系统
- Sqoop：用于与关系型数据库交互

## 安装

### 安装Java

```bash
# 安装
yum -y install java-1.8.0-openjdk java-1.8.0-openjdk-devel

# 检查是否安装成功
java -version
javac -version
rpm -qa | grep java

# 配置环境变量
# /usr/lib/jvm/java-1.8.0-openjdk-1.8.0.222.b03-1.el7.x86_64
JAVA_HOME=$(echo /usr/lib/jvm/java-*-openjdk-*-*); echo $JAVA_HOME
echo export JAVA_HOME=$JAVA_HOME>>~/.bashrc

# 检查环境变量
tail -n 1 ~/.bashrc
ls -l $JAVA_HOME
```

### 展开Hadoop

- **Download [hadoop-*.tar.gz](https://hadoop.apache.org/releases.html)**

```bash
workspace='/usr/local/programs'
if [ ! -e $workspace ]; then mkdir $workspace; fi; cd $workspace
# 上传 hadoop-*.tar.gz 到 $workspace

# 获取名称
hadoop_tar=$(echo ./hadoop-*.tar.gz)
hadoop_file=$(basename ${hadoop_tar})
hadoop_name=${hadoop_file%.*}; hadoop_name=${hadoop_name%.*}

# 展开
tar -zxvf $hadoop_tar -C ./

# 配置环境变量
HADOOP_HOME=$workspace/$hadoop_name
echo export HADOOP_HOME=$HADOOP_HOME>>~/.bashrc

# 检查环境变量
tail -n 1 ~/.bashrc
ls -l $HADOOP_HOME
```

#### 配置守护进程

Hadoop的5个守护进程分别为

- NameNode：管理文件系统名称空间和对集群中存储的文件的访问
- SecondaryNamenode：提供周期检查点和清理任务，一般运行在一台非NameNode的机器上
- DataNode：管理连接到节点的存储（一个集群中可以有多个节点）
- ResourceManager：负责全局的资源管理和任务调度，把整个集群当成计算资源池，只关注分配，不管应用，且不负责容错
- NodeManager：执行在单个节点上的代理，它管理Hadoop集群中单个计算节点

```bash
hadoop_env=$HADOOP_HOME/etc/hadoop/hadoop-env.sh

# 追加以下内容
echo HDFS_NAMENODE_USER=root>>$hadoop_env
echo HDFS_SECONDARYNAMENODE_USER=root>>$hadoop_env
echo HDFS_DATANODE_USER=root>>$hadoop_env
echo YARN_RESOURCEMANAGER_USER=root>>$hadoop_env
echo YARN_NODEMANAGER_USER=root>>$hadoop_env

# 检查结果
tail -n 6 $hadoop_env
unset hadoop_env
```

### PATH

```bash
echo 'export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH;'>>~/.bashrc; source ~/.bashrc

# 检查结果
echo $PATH
tail -n 3 ~/.bashrc
```

### 部署Hadoop

#### 单机部署

仅用于程序调试

```bash
mkdir -p '/tmp/workspace'; cd '/tmp/workspace'
mkdir ./input; cp $HADOOP_HOME/etc/hadoop/core-site.xml ./input

# 单机模式运行MapReduce
example_jar=$(echo $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar); echo $example_jar

hadoop jar $example_jar wordcount input 'dfs[az]+'
```

#### 伪分布部署

相关守护进程都独立运行，但运行于同一台计算机上

```bash
cd $HADOOP_HOME/etc/hadoop
```

```bash
cp ./core-site.xml ./core-site.xml.bk; ll ./core-*
vi ./core-site.xml
```

**core-site.xml**

```xml
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>file:/usr/local/programs/hadoop-3.1.3/tmp</value>
        <description>Default is /tmp/hadoop-`whoami`.</description>
    </property>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
```

```bash
cp ./hdfs-site.xml ./hdfs-site.xml.bk; ll ./hdfs-*
vi ./hdfs-site.xml
```

**hdfs-site.xml**

```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:/usr/local/hadoop/tmp/dfs/name</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:/usr/local/hadoop/tmp/dfs/data</value>
    </property>
</configuration>
```

```bash
# 格式化文件系统
hdfs namenode -format

# /************************************************************
# SHUTDOWN_MSG: Shutting down NameNode at localhost/127.0.0.1
# ************************************************************/

# 启动HDFS
start-dfs.sh

# 关闭防火墙
systemctl stop firewalld

ifconfig | grep 'inet '
# http://<ip>:9870
```

## 常用命令

### 系统管理

```bash
# 启动/停止 HDFS
start-dfs.sh
stop-dfs.sh

# 启动/关闭 YARN
start-yarn.sh
stop-yarn.sh

# 启动/停止 所有进程
start-all.sh
stop-all.sh

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
