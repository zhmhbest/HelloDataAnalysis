
展开`hadoop-*.tar.gz`，将根目录地址添加到环境变量`HADOOP_HOME`、在`PATH`后追加`%HADOOP_HOME%\bin`、`%HADOOP_HOME%\sbin`。

<!-- #### winutils

下载[winutils](https://github.com/cdarlint/winutils)并展开到`%HADOOP_HOME%\bin`。 -->

#### 测试

```batch
hadoop version
```

```txt
Hadoop ?.?.?
```
