# Topic Scaling
# application on SOTU corpus

rm(list=ls()) # cleaning memory

library(lubridate)
library(tidytext)
library(SnowballC)
library(lda)
library(penalized)
library(nnet)
library(tidyverse)
library(quanteda)
library(quanteda.textmodels)
library(tm)

options(stringsAsFactors = F)

# downloading state of the union speeches from qaunteda website
data_corpus_sotu <- readRDS(url("https://quanteda.org/data/data_corpus_sotu.rds"))

# Viewing the dataframe of SOTU
View(data_corpus_sotu[["documents"]])

# creating an index variable id
# and a variable year
data_sotu <- data_corpus_sotu[["documents"]] %>% 
  dplyr::mutate(id=seq.int(nrow(data_corpus_sotu[["documents"]])),
                year=lubridate::year(Date))

# filtering data to keep the period post-1853 (polarity democratic/republican)
data_sotu <- data_sotu %>% 
  dplyr::filter(id>=65) 

data_sotu <- data_sotu%>% 
  dplyr::select(-id) %>%
  dplyr::mutate(id=seq.int(nrow(data_sotu)))

# creating a corpus
corpus_sotu <- corpus(
  data_sotu,
  docid_field = "_document",
  text_field = "texts",
  meta = list(),
  unique_docnames = TRUE)


# dfm transformation
dfmat_sotu <- dfm(corpus_sotu, 
                  remove_punct = TRUE,
                  remove=stopwords("english"),
                  stem = FALSE) %>% dfm_trim(min_termfreq =3)

# Estimating document positions (Wordfish)
tmod_wf <- textmodel_wordfish(dfmat_sotu, dir=c(138,137))

# wordfish visualizations
textplot_scale1d(tmod_wf)

# data manipulation for fancy plots
df <- data.frame(feature = tmod_wf$features, 
                 psi = tmod_wf$psi,
                 beta = tmod_wf$beta)

doclabels <- docnames(tmod_wf$x)

results <- data.frame(doclabels = doclabels, theta = tmod_wf$theta, 
                      lower = tmod_wf$theta - 1.96 * tmod_wf$se.theta, upper = tmod_wf$theta + 
                        1.96 * tmod_wf$se.theta) %>% 
  left_join(dfmat_sotu@docvars,by=c("doclabels"="docname_"))

# density plot for the two parties
results %>% 
  ggplot(aes(x=theta,color=party)) + 
  geom_density() +
  ylab("") + 
  xlab("Psi") +
  theme_bw() +
  theme(legend.position = "top",legend.title = element_blank()) + 
  scale_color_manual(values=c("blue", "red"))

# plotting the whole scale
results %>% 
  ggplot2::ggplot(aes(x = doclabels, y = theta)) +
  geom_point(aes(x = reorder(doclabels, -year), y = theta)) + 
  geom_pointrange(aes(ymin = lower, ymax = upper)) + 
  stat_smooth(method = "loess", aes(group = 1)) +
  coord_flip() + 
  ylab("Psi") + 
  xlab("Address") +
  theme_bw() +
  theme(axis.text=element_text(size=7)) 

# two subplots for parties
ggplot2::ggplot(data = results, aes(x = doclabels, y = theta)) +
  geom_point(aes(x = reorder(doclabels, -year), y = theta)) + 
  geom_pointrange(aes(ymin = lower, ymax = upper)) +
  stat_smooth(method = "loess", aes(group = 1)) +
  facet_grid(docvars(dfmat_sotu, "party")~.,
             scales = "free_y", space="free") +
  coord_flip() + 
  ylab("Psi") + 
  xlab(NULL) +
  theme_bw() +
  theme(axis.text=element_text(size=7))

# pipelines to run sLDA

# removing stopwords ans keeping frequencies >= 3
stopWords <- data.frame(word=stopwords("english")) 

data_m <- unnest_tokens(data_sotu,word,texts) %>% 
  dplyr::anti_join(data.frame(stopWords)) %>% 
  count(word) %>% 
  filter(n>=5)

list_words <- data.frame(word=data_m$word)


data_m <- unnest_tokens(data_sotu,word,texts) %>% 
  dplyr::anti_join(data.frame(stopWords)) %>% 
  dplyr::right_join(list_words) %>%
  group_by(id) %>% 
  summarise(word = paste(word, collapse = " "))

# input for sLDA function
data_m.slda <- data_m %>% 
  pull(word) %>% 
  lexicalize(sep = " ", lower = TRUE, count = 1L, vocab = NULL)

# Estimating sLDA with 4 topics based on Wordfish scores
set.seed(987654321)

num_topics <- 4 # number of topics to be estimated

params <- sample(c(-1, 1), num_topics, replace = TRUE)  ## starting values for sLDA

slda_mod <- slda.em(documents = data_m.slda$documents,
                        K = num_topics, 
                        vocab = data_m.slda$vocab, 
                        num.e.iterations = 50, 
                        num.m.iterations = 20, 
                        alpha = 1, 
                        eta = 0.1, 
                        annotations = tmod_wf$theta,
                        params = params, 
                        variance = var(tmod_wf$theta),
                        logistic=FALSE,
                        method = "sLDA")


# print topic scores
slda_mod$model

# review topics
slda_mod$topics %>% top.topic.words(20, by.score = TRUE)

