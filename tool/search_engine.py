'''
简易搜索引擎
https://blog.csdn.net/ryinlovec/article/details/53547233
'''


'''网页爬虫'''
from urllib import request
# response=request.urlopen('http://www.baidu.com')
# content=response.read().decode('gb18030')
#print(content)#将会print整个网页的html源码


'''爬虫使用BFS算法'''
from collections import deque
# queue=deque(['队列元素1','队列元素2','队列元素3'])
# queue.append('队列元素4')
# queue.popleft()#队首出队
# print(queue)


'''正则表达式'''
import re
#re.match返回一个Match对象
# if re.match(r'href=\".*view\.sdu\.edu\.cn.*\"','href="http://www.view.sdu.edu.cn/new/"'):
#     print('ok')
# else:
#     print('failed')


'''数据保存'''
import sqlite3
# conn=sqlite3.connect('databasetest.db')
# c=conn.cursor()
# #创建一个表
# c.execute('create table doc (id int primary key,link text)')
# #往表格插入一行数据
# num=1
# link='www.baidu.com'
# c.execute('insert into doc values (?,?)',(num,link))
# #查询表格内容
# c.execute('select * from doc')
# #得到查询结果
# result=c.fetchall()
# print(type(result),result)
# conn.commit()
# conn.close()


'''文本匹配'''
# import jieba
# #精确模式，试图将句子最精确地切开，适合文本分析
# seglist=jieba.cut('阴阳师总是抽不到茨木童子好伤心')
# print('/'.join(seglist))
#
# #全模式，把句子中所有的可以成词的词语都扫描出来,速度非常快，但是不能解决歧义
# seglist=jieba.cut('阴阳师总是抽不到茨木童子好伤心',cut_all=True)
# print('/'.join(seglist))
#
# #搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
# seglist=jieba.cut_for_search('阴阳师总是抽不到茨木童子好伤心')
# print('/'.join(seglist))


import sys
from collections import deque
import urllib
from urllib import request
import re
from bs4 import BeautifulSoup
import lxml
import sqlite3
import jieba

# 输入判断
# safelock=input('你确定要重新构建约5000篇文档的词库吗？(y/n)')
# if safelock!='y':
#     sys.exit('终止。')

url='http://www.view.sdu.edu.cn' #入口

queue=deque() #待爬取链接的集合，使用广度优先搜索
visited=set() #已访问的链接集合
queue.append(url)

'''连接数据库表处理'''
conn=sqlite3.connect('viewsdu.db')
c=conn.cursor()
#在create table之前先drop table是因为我之前测试的时候已经建过table了，所以再次运行代码的时候得把旧的table删了重新建
c.execute('DROP TABLE  IF EXISTS doc')
c.execute('create table doc (id int primary key,link text)')
c.execute('DROP TABLE IF EXISTS word')
c.execute('create table word (term varchar(25) primary key,list text)')
conn.commit()
conn.close()


print('***************开始！***************************************************')
cnt=0

while queue:
    url=queue.popleft()
    visited.add(url)
    cnt+=1
    print('开始抓取第',cnt,'个链接：',url)

    #爬取网页内容
    try:
        response=request.urlopen(url)
        content=response.read().decode('utf-8')  #decode('gb18030')
        print(content)
    except:
        continue

    #寻找下一个可爬的链接，因为搜索范围是网站内，所以对链接有格式要求，这个格式要求根据具体情况而定
    m=re.findall(r'<a href=\"([0-9a-zA-Z\_\/\.\%\?\=\-\&]+)\" target=\"_blank\">',content,re.I)
    print(m)
    for x in m:
        if re.match(r'http.+',x):
            if not re.match(r'http\:\/\/www\.view\.sdu\.edu\.cn\/.+',x):
                continue
        elif re.match(r'\/new\/.+',x):
            x='http://www.view.sdu.edu.cn'+x
        else:
            x='http://www.view.sdu.edu.cn/new/'+x
        if (x not in visited) and (x not in queue):
            queue.append(x)

    #解析网页内容,可能有几种情况,这个也是根据这个网站网页的具体情况写的
    soup=BeautifulSoup(content,'lxml')
    title=soup.title
    article=soup.find('div',class_='text_s',id='content')
    author=soup.find('div',class_='text_c')

    if title==None and article==None and author==None:
        print('无内容的页面。')
        continue

    elif article==None and author==None:
        print('只有标题。')
        title=title.text
        title=''.join(title.split())
        article=''
        author=''

    # elif title==None and author==None:
    #   print('只有内容。')
    #   title=''
    #   article=article.get_text("",strip=True)
    #   article=' '.join(article.split())
    #   author=''

    # elif title==None and article==None:
    #   print('只有作者。')
    #   title=''
    #   article=''
    #   author=author.find_next_sibling('div',class_='text_c').get_text("",strip=True)
    #   author=' '.join(author.split())

    # elif title==None:
    #   print('有内容有作者，缺失标题')
    #   title=''
    #   article=article.get_text("",strip=True)
    #   article=' '.join(article.split())
    #   author=author.find_next_sibling('div',class_='text_c').get_text("",strip=True)
    #   author=' '.join(author.split())

    elif article==None:
        print('有标题有作者，缺失内容') #视频新闻
        title=soup.h1.text
        title=''.join(title.split())
        article=''
        author=author.get_text("",strip=True)
        author=''.join(author.split())

    elif author==None:
        print('有标题有内容，缺失作者')
        title=soup.h1.text
        title=''.join(title.split())
        article=article.get_text("",strip=True)
        article=''.join(article.split())
        author=''

    else:
        title=soup.h1.text
        title=''.join(title.split())
        article=article.get_text("",strip=True)
        article=''.join(article.split())
        author=author.find_next_sibling('div',class_='text_c').get_text("",strip=True)
        author=''.join(author.split())

    print('网页标题：',title)

    #提取出的网页内容存在title,article,author三个字符串里，对它们进行中文分词
    seggen=jieba.cut_for_search(title)
    seglist=list(seggen)
    seggen=jieba.cut_for_search(article)
    seglist+=list(seggen)
    seggen=jieba.cut_for_search(author)
    seglist+=list(seggen)

    #数据存储
    conn=sqlite3.connect("viewsdu.db")
    c=conn.cursor()
    c.execute('insert into doc values(?,?)',(cnt,url))

    #对每个分出的词语建立词表
    for word in seglist:
        # print(word)
        #检验看看这个词语是否已存在于数据库
        c.execute('select list from word where term=?',(word,))
        result=c.fetchall()
        #如果不存在
        if len(result)==0:
            docliststr=str(cnt)
            c.execute('insert into word values(?,?)',(word,docliststr))
        #如果已存在
        else:
            docliststr=result[0][0]#得到字符串
            docliststr+=' '+str(cnt)
            c.execute('update word set list=? where term=?',(docliststr,word))

    conn.commit()
    conn.close()
    print('词表建立完毕=======================================================')


'''关键词搜索'''
import re
import urllib
from urllib import request
from collections import deque
from bs4 import BeautifulSoup
import lxml
import sqlite3
import jieba
import math

conn=sqlite3.connect("viewsdu.db")
c=conn.cursor()
c.execute('select count(*) from doc')
N=1+c.fetchall()[0][0]#文档总数
target=input('请输入搜索词：')
seggen=jieba.cut_for_search(target)
score={}#文档号：文档得分
for word in seggen:
    print('得到查询词：',word)
    #计算score
    tf={}#文档号：文档数
    c.execute('select list from word where term=?',(word,))
    result=c.fetchall()
    print(result)
    if len(result)>0:
        doclist=result[0][0]
        doclist=doclist.split(' ')
        doclist=[int(x) for x in doclist] #把字符串转换为元素为int的list
        df=len(set(doclist)) #当前word对应的df数
        idf=math.log(N/df)
        print('idf：',idf)
        for num in doclist:
            if num in tf:
                tf[num]=tf[num]+1
            else:
                tf[num]=1
        #tf统计结束，现在开始计算score
        for num in tf:
            if num in score:
                #如果该num文档已经有分数了，则累加
                score[num]=score[num]+tf[num]*idf
            else:
                score[num]=tf[num]*idf
sortedlist=sorted(score.items(),key=lambda d:d[1],reverse=True)
#print('得分列表',sortedlist)


cnt=0
for num,docscore in sortedlist:
    cnt=cnt+1
    c.execute('select link from doc where id=?',(num,))
    url=c.fetchall()[0][0]
    print(url,'得分：',docscore)

    try:
        response=request.urlopen(url)
        content=response.read().decode('gb18030')
    except:
        print('oops...读取网页出错')
        continue

    soup=BeautifulSoup(content,'lxml')
    title=soup.title
    if title==None:
        print('No title.')
    else:
        title=title.text
        print(title)
    if cnt>20:
        break
if cnt==0:
    print('无搜索结果')










