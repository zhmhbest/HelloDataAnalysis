
#### 安装Java

```bash
# 安装
yum -y install java-1.8.0-openjdk java-1.8.0-openjdk-devel

# 检查是否安装成功
java -version
javac -version
rpm -qa | grep java

# 配置环境变量
echo "">>~/.bashrc
JAVA_HOME=$(echo /usr/lib/jvm/java-*-openjdk-*-*); echo $JAVA_HOME
echo "export JAVA_HOME=$JAVA_HOME">>~/.bashrc
```

#### 安装Hadoop

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
echo "export HADOOP_HOME=$HADOOP_HOME">>~/.bashrc
```

#### 配置PATH

```bash
echo 'export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH;'>>~/.bashrc
source ~/.bashrc
```

#### 测试

```bash
hadoop version
```

```txt
Hadoop ?.?.?
```
