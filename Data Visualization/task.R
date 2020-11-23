library("twitteR")
library("rtweet")
library("wordcloud")
library("plotrix")
library("plotly")
library("SnowballC")
library("ggplot2") # for graph
library("tm") # Corpus
library("qdap") #Freq Term
library("syuzhet") #To Calculate 8 different emotions

#setwd("C:\\Users\\.....")

#consumerKey     <- "mention here"
#consumerSecret  <- "mention here"
#accessToken     <- "mention here"
#accessSecret    <- "mention here"

#setup_twitter_oauth(consumer_key=consumerKey, consumer_secret=consumerSecret,  access_token=accessToken, access_secret=accessSecret)


#tweets <- searchTwitter("#AvengersEndgame OR #avengersendgame OR #movie OR #thenos OR #captainamerica OR #MarvelStudio", n = 150, lang = "en")
#tweets.df <-twListToDF(tweets)
#write.csv(tweets.df, "tweets.csv") 

# time being I haveuploaded dataset By using above code one can directly download data from Twitter developemnet account
tweets.df=read.csv("../input/tweets.csv")

head(tweets.df$text)

# remove Hashtag, URL & Special Character

tweets.df$text=gsub("&amp", "", tweets.df$text)
tweets.df$text = gsub("&amp", "", tweets.df$text)
tweets.df$text = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", tweets.df$text)
tweets.df$text = gsub("@\\w+", "", tweets.df$text)
tweets.df$text = gsub("[[:punct:]]", "", tweets.df$text)
tweets.df$text = gsub("[[:digit:]]", "", tweets.df$text)
tweets.df$text = gsub("http\\w+", "", tweets.df$text)
tweets.df$text = gsub("[ \t]{2,}", "", tweets.df$text)
tweets.df$text = gsub("^\\s+|\\s+$", "", tweets.df$text)

tweets.df$text <- iconv(tweets.df$text, "UTF-8", "ASCII", sub="")
head(tweets.df)

# Emotions for each tweet using NRC dictionary

emotions <- get_nrc_sentiment(tweets.df$text)
emo_bar = colSums(emotions)
emo_sum = data.frame(count_emo=emo_bar, emotion=names(emo_bar))
emo_sum$emotion = factor(emo_sum$emotion,   levels=emo_sum$emotion[order(emo_sum$count_emo,  decreasing = TRUE)])
#emotion.df2 <- cbind(tweets.df, emotion) 
# Visualize the emotions from NRC sentiments

ggplot(emo_sum, aes(x=emo_sum$emotion,y=emo_sum$count_emo,
                    fill=emo_sum$emotion))+
  geom_col(position="stack" )+
  theme()


# Create comparison word cloud data

wordcloud_tweet = c(
  paste(tweets.df$text[emotions$anger > 0], collapse=" "),
  paste(tweets.df$text[emotions$anticipation > 0], collapse=" "),
  paste(tweets.df$text[emotions$disgust > 0], collapse=" "),
  paste(tweets.df$text[emotions$fear > 0], collapse=" "),
  paste(tweets.df$text[emotions$joy > 0], collapse=" "),
  paste(tweets.df$text[emotions$sadness > 0], collapse=" "),
  paste(tweets.df$text[emotions$surprise > 0], collapse=" "),
  paste(tweets.df$text[emotions$trust > 0], collapse=" ")
)

# create corpus
corpus = Corpus(VectorSource(wordcloud_tweet))

# remove punctuation, convert every word in lower case and remove stop words

corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, c(stopwords("english")))
corpus = tm_map(corpus, removeWords,c("also","now","will","guy","there","get","day", "both","just","made", "can","readi","week"))

corpus = tm_map(corpus, stemDocument)


##corpus content
corpus[[3]][1]

##------ Pie Chart ------##
# most frequent terms
term_freq <- freq_terms(corpus, 10)

Review_df <- term_freq %>%
  arrange(desc(WORD)) %>%
  mutate(prop = round(FREQ*100/sum(FREQ), 1),lab.ypos = cumsum(prop) - 0.5*prop)

head(Review_df, 4)

ggplot(Review_df, aes(x = "", y = prop, fill = WORD)) + labs(x = NULL, y = NULL, fill = NULL, title = "Top 10 frequesnt used Words",
                                                             caption="Source: Twitter data")+ geom_bar(width = 8, stat = "identity", color = "white") +  geom_text(aes(y = lab.ypos, label = paste(round(prop), Sep=" %")), 
                                                                                                                                                                   color = "white")+coord_polar("y", start = 2)+ggpubr::fill_palette("default")+  theme_minimal()+  theme(axis.line = element_blank(),
                                                                                                                                                                                                                                                                          axis.text = element_blank(),   axis.ticks = element_blank(),plot.title = element_text(hjust = 0.5, color = "#667866"))

##------ Bar Chart ------##
term_freq <- freq_terms(corpus, 20)
ggplot(term_freq,aes(x=reorder(WORD, -FREQ),y =FREQ,fill = term_freq$WORD)) +  geom_bar(stat = "identity",show.legend = FALSE) +
  theme_minimal()+  geom_text(aes(label = FREQ), vjust = -0.3) 

# create document term matrix
tdm = TermDocumentMatrix(corpus)

# convert as matrix
tdm = as.matrix(tdm)
tdmnew <- tdm[nchar(rownames(tdm)) < 11,]

# Sum rows and frequency data frame
review_term_freq <- rowSums(tdmnew)
# Sort term_frequency in descending order
review_term_freq <- sort(review_term_freq, decreasing = T)
# top 20 most common words
review_term_freq[1:25]

# barchart of the 20 most common words
barplot(review_term_freq[1:25],col = terrain.colors(25),las = 2.5)


####----WORDCLOUD------###
review_word_freq <- data.frame(term = names(review_term_freq),  num = review_term_freq)

# Create a wordcloud for the values in word_freqs
wordcloud(review_word_freq$term, review_word_freq$num, max.words = 100, colors = "brown")

# Print the word cloud with the specified colors
wordcloud(review_word_freq$term, review_word_freq$num, max.words = 100, colors = c("aquamarine","blue","red"))

# column name binding
colnames(tdm) = c('anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust')
colnames(tdmnew) <- colnames(tdm)
comparison.cloud(tdmnew, random.order=FALSE, colors = c("#00B2FF", "red", "#FF0099", "#6600CC", "green", "orange", "blue", "brown"),
                 title.size=1, max.words=1000, scale=c(2.3, 0.4),rot.per=0.4)


# Identify terms shared by both documents
common_words <- subset(tdm, tdm[, 1] > 0 & tdm[, 2] > 0)
# calculate common words and difference
difference <- abs(common_words[, 2] - common_words[, 1])
head(difference)
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[, 3],
                                   decreasing = T), ]
head(common_words)
top25_df <- data.frame(x = common_words[1:25, 1],y = common_words[1:25, 2],  labels = rownames(common_words[1:25, ]))

# pyramid plot n sentiments
pyramid.plot(top25_df$x, top25_df$y,labels = top25_df$labels,
             top.labels=c("Negative","Words","Positive"),
             main="Words in Common",laxlab=NULL,raxlab=NULL, unit = NULL,
             gap=250,space=0.6,  ppmar=c(4,2,4,2),labelcex=0.8,add=FALSE,
             show.values=TRUE,do.first="plot_bg(\"#eddee5\")")+theme_bw()

