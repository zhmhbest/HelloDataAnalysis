<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Hadoop](../index.html)

[TOC]

## 概念

[Hadoop](http://hadoop.apache.org/)是一个由Apache基金会开发的框架，该框架允许使用简单的编程模型**跨计算机集群**对**大型数据集**进行**分布式处理**。其模块包括

- [Hadoop Common](https://www.apache.org/dist/hadoop/common/)
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

- [Download Hadoop Common](https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/)

### Windows安装

@import "install_on_windows.md"

### Linux安装

@import "install_on_linux.md"

## 部署

@import "deploy.md"

### 集群部署

略

## Hadoop命令

@import "command.md"
