import pandas as pd
import numpy as np
import pymysql
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import warnings
import os
import sys
import traceback
import logging
import subprocess

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("bilibili_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 解决OpenGL错误和中文显示问题
matplotlib.use('Agg')  # 使用非GUI后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')

# 数据库连接配置
config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'bilibili_weekly',
    'charset': 'utf8mb4'
}

# 检查Spark环境并设置路径
def setup_spark_environment():
    """配置Spark环境路径"""
    spark_home = None
    
    # 尝试通过环境变量获取Spark路径
    if 'SPARK_HOME' in os.environ:
        spark_home = os.environ['SPARK_HOME']
        logger.info(f"从环境变量找到SPARK_HOME: {spark_home}")
    
    # 尝试常见路径查找Spark
    common_spark_paths = [
        '/usr/local/spark',
        '/opt/spark',
        '/usr/lib/spark',
        '/usr/hdp/current/spark2-client'  # HDP常见路径
    ]
    
    for path in common_spark_paths:
        if os.path.exists(os.path.join(path, 'bin', 'spark-submit')):
            spark_home = path
            logger.info(f"在常见路径找到Spark: {spark_home}")
            break
    
    if spark_home:
        # 添加Spark到系统路径
        spark_python = os.path.join(spark_home, 'python')
        py4j_path = os.path.join(spark_home, 'python', 'lib', 'py4j-*-src.zip')
        
        if os.path.exists(spark_python):
            sys.path.insert(0, spark_python)
        
        py4j_files = [f for f in os.listdir(os.path.join(spark_home, 'python', 'lib')) 
                      if f.startswith('py4j') and f.endswith('.zip')]
        
        if py4j_files:
            py4j_path = os.path.join(spark_home, 'python', 'lib', py4j_files[0])
            sys.path.insert(0, py4j_path)
            logger.info(f"添加Py4J路径: {py4j_path}")
    
    return spark_home

# 尝试设置Spark环境
SPARK_ENABLED = False
SPARK_DF_TYPE = None  # 新增：Spark DataFrame类型标识
spark_home = setup_spark_environment()

# 尝试导入PySpark
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame as SparkDataFrame  # 新增：导入DataFrame类型
    from pyspark.sql import functions as F
    from pyspark import SparkConf
    SPARK_ENABLED = True
    SPARK_DF_TYPE = SparkDataFrame  # 设置类型标识
    logger.info("PySpark 已成功导入")
except ImportError as e:
    logger.warning(f"PySpark 导入失败: {e}")
    SPARK_ENABLED = False

def init_spark():
    """初始化Spark会话（如果可用）"""
    if not SPARK_ENABLED:
        logger.warning("Spark 不可用，跳过初始化")
        return None
    
    try:
        # 创建Spark配置
        conf = SparkConf()
        
        # 基础配置
        conf.setAppName("BilibiliDataAnalysis")
        conf.set("spark.driver.memory", "2g")
        conf.set("spark.executor.memory", "2g")
        conf.set("spark.sql.shuffle.partitions", "4")
        conf.set("spark.logConf", "true")
        
        # 添加MySQL JDBC驱动
        mysql_driver_path = find_mysql_driver()
        if mysql_driver_path:
            # 同时设置spark.jars和spark.driver.extraClassPath
            conf.set("spark.jars", mysql_driver_path)
            conf.set("spark.driver.extraClassPath", mysql_driver_path)
            logger.info(f"添加MySQL驱动: {mysql_driver_path}")
        else:
            logger.warning("未找到MySQL JDBC驱动，Spark可能无法连接数据库")
        
        # 创建Spark会话
        spark = SparkSession.builder \
            .config(conf=conf) \
            .getOrCreate()
        
        # 配置日志级别
        spark.sparkContext.setLogLevel("WARN")
        
        # 打印Spark配置信息
        logger.info("Spark 配置信息:")
        for (k, v) in spark.sparkContext.getConf().getAll():
            logger.info(f"{k} = {v}")
        
        logger.info("Spark 会话初始化成功!")
        return spark
    except Exception as e:
        logger.error(f"初始化 Spark 时出错: {e}")
        traceback.print_exc()
        return None

def find_mysql_driver():
    """查找MySQL JDBC驱动"""
    # 常见驱动路径
    common_driver_paths = [
        '/usr/share/java/mysql-connector-java.jar',  # Ubuntu常见路径
        '/usr/lib/spark/jars/mysql-connector-java.jar',
        '/opt/spark/jars/mysql-connector-java.jar',
        '/home/hadoop/jars/mysql-connector-java.jar',
        '/usr/hdp/current/spark2-client/jars/mysql-connector-java.jar',
        '/home/hadoop/Downloads/mysql-connector-java-8.0.30.jar'  # 您使用的路径
    ]
    
    # 检查驱动是否存在
    for path in common_driver_paths:
        if os.path.exists(path):
            return path
    
    # 尝试在系统中查找
    try:
        find_cmd = "find / -name 'mysql-connector-java*.jar' 2>/dev/null | head -1"
        result = subprocess.run(
            find_cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    
    return None

def create_database():
    """创建数据库和表结构"""
    try:
        # 连接到MySQL服务器
        conn = pymysql.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            charset=config['charset']
        )
        
        cursor = conn.cursor()
        
        # 创建数据库
        cursor.execute("DROP DATABASE IF EXISTS bilibili_weekly")
        cursor.execute("CREATE DATABASE bilibili_weekly CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        
        # 使用数据库
        cursor.execute("USE bilibili_weekly")
        
        # 创建video_categories表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_categories (
            tid INT NOT NULL COMMENT '分区ID',
            tname VARCHAR(50) NOT NULL COMMENT '分区名称',
            PRIMARY KEY (tid),
            UNIQUE KEY (tname)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        
        # 创建weekly_videos表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS weekly_videos (
            aid BIGINT NOT NULL COMMENT '视频ID',
            bvid VARCHAR(20) DEFAULT NULL COMMENT 'BV号',
            title VARCHAR(255) DEFAULT NULL COMMENT '视频标题',
            tname VARCHAR(50) DEFAULT NULL COMMENT '分区名称',
            owner_mid BIGINT DEFAULT NULL COMMENT 'UP主ID',
            owner_name VARCHAR(100) DEFAULT NULL COMMENT 'UP主名称',
            pubdate BIGINT DEFAULT NULL COMMENT '发布时间戳',
            duration INT DEFAULT NULL COMMENT '视频时长(秒)',
            description TEXT COMMENT '视频描述',
            dynamic TEXT COMMENT '动态描述',
            pic_url VARCHAR(255) DEFAULT NULL COMMENT '封面图URL',
            view_count BIGINT DEFAULT NULL COMMENT '播放量',
            danmaku_count INT DEFAULT NULL COMMENT '弹幕数',
            reply_count INT DEFAULT NULL COMMENT '评论数',
            favorite_count INT DEFAULT NULL COMMENT '收藏数',
            coin_count INT DEFAULT NULL COMMENT '硬币数',
            share_count INT DEFAULT NULL COMMENT '分享数',
            like_count INT DEFAULT NULL COMMENT '点赞数',
            cid BIGINT DEFAULT NULL COMMENT '视频CID',
            dimension_width INT DEFAULT NULL COMMENT '视频宽度',
            dimension_height INT DEFAULT NULL COMMENT '视频高度',
            short_link VARCHAR(100) DEFAULT NULL COMMENT '短链接',
            rcmd_reason VARCHAR(255) DEFAULT NULL COMMENT '推荐理由',
            insert_time TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '插入时间',
            PRIMARY KEY (aid)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        
        logger.info("数据库和表结构创建成功！")
        
        # 提交更改并关闭连接
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"创建数据库时出错: {e}")
        traceback.print_exc()

def import_data():
    """从SQL文件导入数据"""
    try:
        # 连接到数据库
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        
        # 读取SQL文件
        sql_file = 'bilibili_weekly.sql'
        if not os.path.exists(sql_file):
            logger.error(f"SQL文件不存在: {sql_file}")
            return
            
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # 分割SQL语句并执行
        sql_commands = sql_content.split(';')
        for command in sql_commands:
            if command.strip():
                try:
                    cursor.execute(command)
                except Exception as e:
                    logger.warning(f"执行SQL时出错: {e}\nSQL: {command[:100]}...")
        
        logger.info("数据导入成功！")
        
        # 提交更改并关闭连接
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"导入数据时出错: {e}")
        traceback.print_exc()

def fetch_data(spark=None):
    """从数据库获取数据（使用Spark或Pandas）"""
    try:
        # 如果Spark可用且初始化成功，使用Spark获取数据
        if spark and SPARK_ENABLED:
            logger.info("使用Spark获取数据...")
            try:
                # 定义Spark JDBC连接参数
                jdbc_url = f"jdbc:mysql://{config['host']}:3306/{config['database']}"
                jdbc_properties = {
                    "user": config['user'],
                    "password": config['password'],
                    "driver": "com.mysql.cj.jdbc.Driver"
                }
                
                # 读取数据 - 使用更可靠的连接方式
                df = spark.read \
                    .format("jdbc") \
                    .option("url", jdbc_url) \
                    .option("dbtable", "(SELECT * FROM weekly_videos) as videos") \
                    .option("user", jdbc_properties["user"]) \
                    .option("password", jdbc_properties["password"]) \
                    .option("driver", "com.mysql.cj.jdbc.Driver") \
                    .load()
                
                # 转换数据类型
                df = df.withColumn("pubdate", F.from_unixtime(F.col("pubdate")).cast("timestamp"))
                df = df.withColumn("duration_min", F.col("duration") / 60)
                
                logger.info(f"成功获取 {df.count()} 条数据 (Spark)")
                return df
            except Exception as e:
                logger.error(f"使用Spark获取数据失败: {e}")
                traceback.print_exc()
                # 继续执行Pandas部分
        
        # 使用Pandas获取数据
        logger.info("使用Pandas获取数据...")
        conn = pymysql.connect(**config)
        query = """
        SELECT 
            wv.aid, wv.title, wv.tname, wv.owner_name, 
            FROM_UNIXTIME(wv.pubdate) AS pubdate,
            wv.duration, wv.view_count, wv.danmaku_count, 
            wv.reply_count, wv.favorite_count, wv.coin_count,
            wv.share_count, wv.like_count
        FROM weekly_videos wv
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        # 转换数据类型
        df['pubdate'] = pd.to_datetime(df['pubdate'])
        df['duration_min'] = df['duration'] / 60  # 转换为分钟
        
        logger.info(f"成功获取 {len(df)} 条数据 (Pandas)")
        return df
    
    except Exception as e:
        logger.error(f"获取数据时出错: {e}")
        traceback.print_exc()
        return None

def analyze_data(data, spark=None):
    """数据分析与可视化（兼容Spark和Pandas）"""
    if data is None:
        logger.warning("没有数据可分析")
        return None
    
    # 检查数据是否为空（使用SPARK_DF_TYPE进行类型检查）
    if SPARK_ENABLED and spark and SPARK_DF_TYPE and isinstance(data, SPARK_DF_TYPE):
        if data.count() == 0:
            logger.warning("没有数据可分析")
            return None
    elif hasattr(data, 'empty') and data.empty:
        logger.warning("没有数据可分析")
        return None
    
    # 确保输出目录存在
    os.makedirs('analysis_results', exist_ok=True)
    
    # 如果使用Spark且数据是Spark DataFrame
    if SPARK_ENABLED and spark and SPARK_DF_TYPE and isinstance(data, SPARK_DF_TYPE):
        logger.info("使用Spark进行数据分析...")
        return analyze_with_spark(data, spark)
    
    # 否则使用Pandas
    else:
        logger.info("使用Pandas进行数据分析...")
        return analyze_with_pandas(data)

def analyze_with_pandas(df):
    """使用Pandas进行数据分析与可视化"""
    # 1. 分区视频数量分布（饼图）
    plt.figure(figsize=(10, 8))
    category_counts = df['tname'].value_counts()
    category_counts.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('视频分区分布', fontsize=14)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('analysis_results/category_distribution.png', dpi=300)
    plt.close()
    
    # 2. 分区平均播放量（柱状图）
    plt.figure(figsize=(12, 8))
    category_views = df.groupby('tname')['view_count'].mean().sort_values(ascending=False)
    sns.barplot(x=category_views.values, y=category_views.index, palette='viridis')
    plt.title('各分区平均播放量', fontsize=14)
    plt.xlabel('平均播放量', fontsize=12)
    plt.ylabel('分区名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('analysis_results/average_views_by_category.png', dpi=300)
    plt.close()
    
    # 3. 播放量与互动指标关系
    plt.figure(figsize=(12, 10))
    
    # 播放量与点赞量
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='view_count', y='like_count', hue='tname', s=100)
    plt.title('播放量与点赞量关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    # 播放量与评论数
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='view_count', y='reply_count', hue='tname', s=100)
    plt.title('播放量与评论数关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    # 播放量与弹幕数
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='view_count', y='danmaku_count', hue='tname', s=100)
    plt.title('播放量与弹幕数关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    # 播放量与分享数
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='view_count', y='share_count', hue='tname', s=100)
    plt.title('播放量与分享数关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('analysis_results/view_interaction_relationships.png', dpi=300)
    plt.close()
    
    # 4. 视频时长分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_min'], bins=20, kde=True)
    plt.title('视频时长分布', fontsize=14)
    plt.xlabel('时长(分钟)', fontsize=12)
    plt.ylabel('视频数量', fontsize=12)
    plt.tight_layout()
    plt.savefig('analysis_results/duration_distribution.png', dpi=300)
    plt.close()
    
    # 5. 时间序列分析（按天统计）
    df['pubdate_day'] = df['pubdate'].dt.date
    daily_stats = df.groupby('pubdate_day').agg({
        'view_count': 'sum',
        'like_count': 'sum',
        'aid': 'count'
    }).rename(columns={'aid': 'video_count'})
    
    # 视频发布趋势
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    daily_stats['video_count'].plot(marker='o', linewidth=2, markersize=8)
    plt.title('每日视频发布数量', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('视频数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 播放量趋势
    plt.subplot(1, 2, 2)
    daily_stats['view_count'].plot(marker='o', color='orange', linewidth=2, markersize=8)
    plt.title('每日总播放量', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('播放量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('analysis_results/daily_trends.png', dpi=300)
    plt.close()
    
    # 6. UP主贡献分析
    top_up = df.groupby('owner_name').agg({
        'view_count': 'sum',
        'aid': 'count'
    }).sort_values('view_count', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    top_up['aid'].plot(kind='bar', color='skyblue')
    plt.title('TOP10 UP主视频数量', fontsize=14)
    plt.xlabel('UP主', fontsize=12)
    plt.ylabel('视频数量', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    top_up['view_count'].plot(kind='bar', color='salmon')
    plt.title('TOP10 UP主播放总量', fontsize=14)
    plt.xlabel('UP主', fontsize=12)
    plt.ylabel('播放总量', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('analysis_results/top_up_analysis.png', dpi=300)
    plt.close()
    
    return daily_stats

def analyze_with_spark(spark_df, spark):
    """使用Spark进行数据分析与可视化"""
    # 1. 分区视频数量分布（饼图）
    category_counts = spark_df.groupBy("tname").count().toPandas()
    plt.figure(figsize=(10, 8))
    plt.pie(category_counts['count'], labels=category_counts['tname'], autopct='%1.1f%%', startangle=90)
    plt.title('视频分区分布', fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis_results/category_distribution.png', dpi=300)
    plt.close()
    
    # 2. 分区平均播放量（柱状图）
    category_views = spark_df.groupBy("tname").agg(F.avg("view_count").alias("avg_view_count"))
    category_views_pd = category_views.orderBy(F.desc("avg_view_count")).toPandas()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='avg_view_count', y='tname', data=category_views_pd, palette='viridis')
    plt.title('各分区平均播放量', fontsize=14)
    plt.xlabel('平均播放量', fontsize=12)
    plt.ylabel('分区名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('analysis_results/average_views_by_category.png', dpi=300)
    plt.close()
    
    # 3. 播放量与互动指标关系（使用Spark SQL）
    spark_df.createOrReplaceTempView("videos")
    
    # 播放量与点赞量
    like_df = spark.sql("""
        SELECT view_count, like_count, tname 
        FROM videos
        WHERE view_count > 0 AND like_count > 0
    """).toPandas()
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=like_df, x='view_count', y='like_count', hue='tname', s=100)
    plt.title('播放量与点赞量关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    # 播放量与评论数
    reply_df = spark.sql("""
        SELECT view_count, reply_count, tname 
        FROM videos
        WHERE view_count > 0 AND reply_count > 0
    """).toPandas()
    
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=reply_df, x='view_count', y='reply_count', hue='tname', s=100)
    plt.title('播放量与评论数关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    # 播放量与弹幕数
    danmaku_df = spark.sql("""
        SELECT view_count, danmaku_count, tname 
        FROM videos
        WHERE view_count > 0 AND danmaku_count > 0
    """).toPandas()
    
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=danmaku_df, x='view_count', y='danmaku_count', hue='tname', s=100)
    plt.title('播放量与弹幕数关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    # 播放量与分享数
    share_df = spark.sql("""
        SELECT view_count, share_count, tname 
        FROM videos
        WHERE view_count > 0 AND share_count > 0
    """).toPandas()
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=share_df, x='view_count', y='share_count', hue='tname', s=100)
    plt.title('播放量与分享数关系', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('analysis_results/view_interaction_relationships.png', dpi=300)
    plt.close()
    
    # 4. 视频时长分布
    duration_df = spark_df.select("duration_min").toPandas()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(duration_df['duration_min'], bins=20, kde=True)
    plt.title('视频时长分布', fontsize=14)
    plt.xlabel('时长(分钟)', fontsize=12)
    plt.ylabel('视频数量', fontsize=12)
    plt.tight_layout()
    plt.savefig('analysis_results/duration_distribution.png', dpi=300)
    plt.close()
    
    # 5. 时间序列分析（按天统计）
    daily_stats = spark_df.withColumn("pubdate_day", F.date_format("pubdate", "yyyy-MM-dd")) \
        .groupBy("pubdate_day") \
        .agg(
            F.sum("view_count").alias("view_count"),
            F.sum("like_count").alias("like_count"),
            F.count("aid").alias("video_count")
        )
    
    daily_stats_pd = daily_stats.toPandas()
    daily_stats_pd['pubdate_day'] = pd.to_datetime(daily_stats_pd['pubdate_day'])
    daily_stats_pd = daily_stats_pd.set_index('pubdate_day').sort_index()
    
    # 视频发布趋势
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    daily_stats_pd['video_count'].plot(marker='o', linewidth=2, markersize=8)
    plt.title('每日视频发布数量', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('视频数量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 播放量趋势
    plt.subplot(1, 2, 2)
    daily_stats_pd['view_count'].plot(marker='o', color='orange', linewidth=2, markersize=8)
    plt.title('每日总播放量', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('播放量', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('analysis_results/daily_trends.png', dpi=300)
    plt.close()
    
    # 6. UP主贡献分析
    top_up = spark_df.groupBy("owner_name") \
        .agg(
            F.sum("view_count").alias("total_views"),
            F.count("aid").alias("video_count")
        ) \
        .orderBy(F.desc("total_views")) \
        .limit(10) \
        .toPandas()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x='video_count', y='owner_name', data=top_up, color='skyblue')
    plt.title('TOP10 UP主视频数量', fontsize=14)
    plt.xlabel('视频数量', fontsize=12)
    plt.ylabel('UP主', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='total_views', y='owner_name', data=top_up, color='salmon')
    plt.title('TOP10 UP主播放总量', fontsize=14)
    plt.xlabel('播放总量', fontsize=12)
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('analysis_results/top_up_analysis.png', dpi=300)
    plt.close()
    
    return daily_stats_pd

def predict_trends(daily_stats):
    """趋势预测"""
    if daily_stats is None or daily_stats.empty:
        logger.warning("数据不足或没有数据可预测")
        return pd.DataFrame()
    
    # 准备数据
    X = np.array(range(len(daily_stats))).reshape(-1, 1)
    y_view = daily_stats['view_count'].values
    
    # 多项式特征转换（2次多项式）
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # 训练模型
    model_view = LinearRegression()
    model_view.fit(X_poly, y_view)
    
    # 预测未来3天
    future_days = np.array(range(len(daily_stats), len(daily_stats) + 3)).reshape(-1, 1)
    future_days_poly = poly.transform(future_days)
    predictions = model_view.predict(future_days_poly)
    
    # 计算R²分数
    y_pred = model_view.predict(X_poly)
    r2 = r2_score(y_view, y_pred)
    
    # 生成未来日期
    last_date = daily_stats.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(3)]
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    
    # 历史数据
    plt.plot(daily_stats.index, y_view, 'o-', label='实际播放量', markersize=8, linewidth=2)
    plt.plot(daily_stats.index, y_pred, 'r--', label='拟合曲线', linewidth=2)
    
    # 预测数据
    plt.plot(future_dates, predictions, 'go-', markersize=8, label='预测播放量', linewidth=2)
    
    # 添加预测值标签
    for i, pred in enumerate(predictions):
        plt.text(future_dates[i], pred, f'{pred/10000:.1f}万', 
                 ha='center', va='bottom', fontsize=10)
    
    plt.title(f'播放量趋势预测 (R²={r2:.2f})', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('播放量', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_results/view_predictions.png', dpi=300)
    plt.close()
    
    # 返回预测结果
    prediction_df = pd.DataFrame({
        '预测日期': future_dates,
        '预测播放量': predictions,
        '预测播放量(万)': [f'{p/10000:.1f}万' for p in predictions]
    })
    
    return prediction_df

def main():
    logger.info("="*50)
    logger.info("Bilibili周数据分析系统")
    logger.info("="*50)
    
    spark = None
    if SPARK_ENABLED:
        logger.info("\n[步骤0] 初始化Spark会话...")
        spark = init_spark()
    
    # 创建数据库和表结构
    logger.info("\n[步骤1] 创建数据库...")
    create_database()
    
    # 导入数据
    logger.info("\n[步骤2] 导入数据...")
    import_data()
    
    # 获取数据
    logger.info("\n[步骤3] 获取数据...")
    data = fetch_data(spark)
    
    if data is not None:
        # 分析数据并可视化
        logger.info("\n[步骤4] 数据分析与可视化...")
        daily_stats = analyze_data(data, spark)
        
        # 趋势预测
        logger.info("\n[步骤5] 趋势预测...")
        if daily_stats is not None and not daily_stats.empty:
            prediction_df = predict_trends(daily_stats)
            logger.info("\n未来播放量预测:")
            logger.info(prediction_df[['预测日期', '预测播放量(万)']].to_string(index=False))
        
        logger.info("\n分析完成! 所有图表已保存至 'analysis_results' 目录")
    else:
        logger.error("\n分析失败: 没有获取到数据")
    
    # 停止Spark会话（如果存在）
    if spark:
        logger.info("\n[步骤6] 停止Spark会话...")
        spark.stop()
        logger.info("Spark会话已停止")

if __name__ == "__main__":
    main()
