#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# In[2]:


comments=pd.read_csv(r'C:\Users\Pratiksha Singh\Desktop\real project dataanaly\YOUTUBE/UScomments.csv',error_bad_lines=False)


# In[3]:


comments.head() #to get first five rows


# In[4]:


#finding missing values total count


# In[5]:


comments.isnull().sum()


# In[6]:


#dropping the missing value


# In[7]:


comments.dropna(inplace=True)


# In[8]:


comments.isnull().sum()


# In[9]:


#performing sentiment analysis


# In[10]:


get_ipython().system('pip install textblob')


# In[11]:


from textblob import TextBlob


# In[12]:


comments.head(6)


# In[13]:


TextBlob("Say hi to Kong and maverick for me	").sentiment.polarity


# In[14]:


comments.shape


# In[15]:


#now we have to do iteration



# In[ ]:





# In[16]:


polarity=[]
for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)
    
    


# In[17]:


len(polarity)


# In[18]:


comments['polarity']=polarity
comments.head(5)


# In[19]:


#performing word cloud analysis for positive polarity.


# In[20]:


filter1 = comments['polarity']==1


# In[21]:


comments_positive = comments[filter1]


# In[22]:


filter2 = comments['polarity']==-1


# In[23]:


comments_negative = comments[filter2]


# In[24]:


comments_negative.head(5)


# In[25]:


comments_positive.head(5)


# In[26]:


get_ipython().system('pip install wordcloud')


# In[27]:


from wordcloud import WordCloud , STOPWORDS


# In[28]:


set(STOPWORDS)


# In[29]:


comments['comment_text']


# In[30]:


type(comments['comment_text'])


# In[31]:


total_comments_positive = ' '.join(comments_positive['comment_text'])


# In[32]:


wordcloud=WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)


# In[33]:


plt.imshow(wordcloud)
plt.axis('off')


# In[34]:


total_comments_negative = ' '.join(comments_negative['comment_text'])


# In[35]:


wordcloud2 = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)


# In[36]:


plt.imshow(wordcloud2)
plt.axis('off')


# In[37]:


#Emoji Analysis


# In[38]:


get_ipython().system('pip install emoji==2.2.0')


# In[39]:


import emoji


# In[40]:


comments['comment_text'].head(6)


# In[41]:


comment = 'trending 😉'


# In[42]:


emoji_list = []

for char in comment:
    if char in emoji.EMOJI_DATA:
        emoji_list.append(char)


# In[43]:


emoji_list


# In[44]:


all_emojis_list = []

for comment in comments['comment_text'].dropna(): ## in case u have missing values , call dropna()
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emojis_list.append(char)


# In[45]:


all_emojis_list[0:10]


# In[46]:


from collections import Counter


# In[47]:


Counter(all_emojis_list).most_common(10)


# In[48]:


Counter(all_emojis_list).most_common(10)[0]


# In[49]:


Counter(all_emojis_list).most_common(10)[0][0]


# In[50]:


Counter(all_emojis_list).most_common(10)[1][0]


# In[51]:


Counter(all_emojis_list).most_common(10)[2][0]


# In[52]:


emojis = [Counter(all_emojis_list).most_common(10)[i][0] for i in range(10)]


# In[53]:


Counter(all_emojis_list).most_common(10)[2][1]


# In[54]:


freqs = [Counter(all_emojis_list).most_common(10)[i][1] for i in range(10)]


# In[55]:


freqs


# In[56]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[57]:


trace = go.Bar(x=emojis , y=freqs)


# In[58]:


iplot([trace])


# In[59]:


#collect entire data of youtube,data collection


# In[60]:


import os


# In[62]:


files=os.listdir(r'C:\Users\Pratiksha Singh\Desktop\real project dataanaly\YOUTUBE\additional_data')


# In[64]:


files


# In[65]:


## extracting csv files only from above list

files_csv = [file for file in files if '.csv' in file]


# In[66]:


files_csv


# In[68]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[69]:


full_df = pd.DataFrame()
path = r'C:\Users\Pratiksha Singh\Desktop\real project dataanaly\YOUTUBE\additional_data'


for file in files_csv:
    current_df = pd.read_csv(path+'/'+file , encoding='iso-8859-1' , error_bad_lines=False)
    
    full_df = pd.concat([full_df , current_df] , ignore_index=True)


# In[70]:


full_df.shape


# In[71]:


full_df[full_df.duplicated()].shape


# In[72]:


full_df = full_df.drop_duplicates()


# In[73]:


full_df.shape


# In[75]:


full_df.to_csv(r'C:\Users\Pratiksha Singh\Desktop\real project dataanaly\YOUTUBE\New folder\youtubedataexport.csv' , index=False)


# In[76]:


#create engine allows us to connect to database
from sqlalchemy import create_engine


# In[77]:


# Lets create sql_alchemy engine by using create_engine method ie create engine allows us to connect to database
engine = create_engine(r'sqlite:///C:\Users\Pratiksha Singh\Desktop\real project dataanaly\YOUTUBE\New folder\youtubedataexport.sqlite')


# In[78]:


full_df.to_sql('Users' , con=engine , if_exists='append')


# In[79]:


# As soon as u have u have your data into 'youtubedataexport.sqlite' which has table has 'Users', now u can read data from this db file 'youtube_whole_data.sqlite' using sqlite3 & pandas


# In[80]:


#Analysing most liked category


# In[81]:


full_df.head(5)


# In[82]:


full_df['category_id'].unique()


# In[83]:


## lets read json file ..
json_df = pd.read_json(r'C:\Users\Pratiksha Singh\Desktop\real project dataanaly\YOUTUBE\additional_data/US_category_id.json')


# In[84]:


json_df


# In[85]:


json_df['items'][0]


# In[86]:


json_df['items'][1]


# In[92]:


cat_dict={}
for item in json_df['items'].values:
    cat_dict[int(item['id'])]=item['snippet']['title']


# In[93]:


cat_dict


# In[94]:


full_df['category_name'] = full_df['category_id'].map(cat_dict)


# In[95]:


full_df.head(4)


# In[96]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name' , y='likes' , data=full_df)
plt.xticks(rotation='vertical')


# In[97]:


#to find out whether audience are engaged or not.


# In[98]:


full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislike_rate'] = (full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[99]:


full_df.columns


# In[100]:


plt.figure(figsize=(8,6))
sns.boxplot(x='category_name' , y='like_rate' , data=full_df)
plt.xticks(rotation='vertical')
plt.show()


# In[101]:


### analysing relationship between views & likes


# In[102]:


sns.regplot(x='views' , y='likes' , data = full_df)


# In[103]:


full_df.columns


# In[104]:


full_df[['views', 'likes', 'dislikes']].corr() ### finding co-relation values between ['views', 'likes', 'dislikes']


# In[109]:


sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr(),annot=True)


# In[121]:


#analysing trending videos of youtube.


# In[122]:


full_df.head(6)


# In[123]:


full_df['channel_title'].value_counts()


# In[124]:


cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[125]:


cdf = cdf.rename(columns={0:'total_videos'})


# In[126]:


cdf


# In[127]:


import plotly.express as px
px.bar(data_frame=cdf[0:20] , x='channel_title' , y='total_videos')


# In[128]:


#does punctuation has an impact on likes, dislikes and views.


# In[111]:


full_df['title'][0]


# In[112]:


import string


# In[113]:


string.punctuation


# In[114]:


len([char for char in full_df['title'][0] if char in string.punctuation])


# In[115]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[116]:


sample = full_df[0:10000]


# In[117]:


sample['count_punc'] = sample['title'].apply(punc_count)


# In[118]:


sample['count_punc']


# In[119]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='views' , data=sample)
plt.show()


# In[120]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='likes' , data=sample)
plt.show()


# In[ ]:




