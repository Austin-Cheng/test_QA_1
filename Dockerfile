# 基于镜像基础
FROM python_basic:3.9

# 设置代码文件夹工作目录 /app
WORKDIR /app
# 复制当前代码文件到容器中 /app
ADD . /app
# 安装依赖
# RUN pipenv install

ENV PORT=8080
# 项目名称
ENV PROJECT_NAME=it_maintenance_search
ENV NEBULA_IPS=10.4.131.25
ENV NEBULA_PORTS=9669
ENV NEBULA_USER=root
ENV NEBULA_PASSWORD=root
# 算法配置
# 用于bert词向量相似度计算
ENV BERT_SERVER_IP=10.4.15.191

EXPOSE 8080

# 执行命令
CMD python main.py
