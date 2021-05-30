# title: Decision Trees and Random Forests to Predict the Price Range
# of Mobile Phones
# author: "Alejandro Zarazo"


#Install missing required packages
if(!require(lubridate)) install.packages("lubridate")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret") 
if(!require(readr)) install.packages("readr")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(GGally)) install.packages("GGally")
if(!require(rpart)) install.packages("rpart")   #Decision Tree
if(!require(rpart.plot))  ("rpart.plot")     #Decision Tree plot
if(!require(dplyr)) install.packages("dplyr") 
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(randomForest)) install.packages("randomForest")


# 2. Dataset

# Downloading and charging the dataset
url <- "https://github.com/drrueda/DataSets/archive/refs/heads/main.zip"
download.file(url,"temp.zip", mode="wb")
unzip_result <- unzip("temp.zip", exdir = "data", overwrite = TRUE)
mobile <- read.csv(unzip_result)

# Description of the data

#Display the head of the mobile dataset
head(mobile,6)


# 3. Methods and Analysis
# 3.1. Data Exploration, Cleaning and Visualization
# 3.1.1. Exploration

#  To explore the variables
glimpse(mobile)

# Transform the variables that are categorical
mobile <- mobile %>%
  mutate(
    blue = as.factor(blue), 
    dual_sim  = as.factor(dual_sim),
    four_g = as.factor(four_g),
    three_g = as.factor(three_g),
    touch_screen = as.factor(touch_screen),
    wifi = as.factor(wifi),
    price_range = as.factor(price_range),
    # Categorical values that the price variable can assume
    price_range = sapply(price_range, 
                         switch,"low cost","medium cost", 
                         "high cost","very high cost"),
    # Order in which the price range must appear
    price_range = ordered(price_range,
                          levels=c("low cost","medium cost", "high cost","very high cost"))
  )
# We see how the variables are after transforming to factor
str(mobile)

# 3.1.2. Data Tidying

# Missing values by column
colSums(is.na(mobile))

## 3.1.3. Visualization

# Graph to see the relation of RAM memory vs Price
ggplot(mobile, aes(price_range, ram)) + 
  geom_point() +
  geom_rug(size=0.1) +   
  theme_set(theme_minimal(base_size = 18))+
  ylab('RAM Memory of the Device')+
  xlab('Price Segment')

# We group and count if 4G support is included or not
mobile %>% group_by(four_g) %>% summarise(freq=n()) %>% 
  # Pie chart to determine the percentage of phones with 4G support 
  ggplot( aes(x="", y=freq, fill=four_g)) + 
  geom_bar(stat="identity", width=1)+
  coord_polar("y", start=0) + 
  # We obtain the percentage
  geom_text(aes(label = paste0(round((freq/sum(freq))*100), "%")),
            position = position_stack(vjust = 0.5),color="white")+
  # We apply color
  scale_fill_manual(values=c("#FF0099","#0066FF"))+
  # We include the legend
  labs(x = NULL, y = NULL, 
       fill = "0 = Without 4G Support / 1 = With 4G Support", title = "Percentage of 4G Support")+
  theme_classic() + 
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=9,face="bold"), 
        legend.position = "right"
  )

# We group by price range
mobile %>% group_by(price_range) %>% summarise(freq=n()) %>%
  # We obtain the frequency by price
  ggplot( aes(x="", y=freq, fill=price_range)) +
  geom_bar(stat="identity", width=1)+
  coord_polar("y", start=0) + 
  # We obtain the percentage by price
  geom_text(aes(label = paste0(round((freq/sum(freq))*100), "%")),
            position = position_stack(vjust = 0.5),color="white")+
  scale_fill_manual(values=c("#FF0099", # We apply some color
                             "#0066FF","#FF9900","#00FF66")) +
  labs(x = NULL, y = NULL, fill = "Price Range", 
       title = "Proportion Range/Price")+
  theme_classic() + 
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=9,face="bold"), 
        legend.position = "right"
  )

ggcorr(mobile,label = T, size=3, 
       label_size = 3, hjust=0.95, 
       # Color for the variables with the highest positive correlation
       layout.exp = 3,low = "#0066FF",
       high = "#FF9900")+ # Variables with lower correlation
  labs(
    title="Correlation Matrix"
  )+
  theme_minimal()+
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.title=element_text(size=8,face="bold"), 
    axis.text.y=element_blank()
  )

# 3.2. Modeling Approaches
# 3.2.1. Decision Tree

set.seed(1600) # To permit the reproduction of the results
# 70% of the dataset for training (random sample). 
# We avoid overtraining.
x_train <-sample_frac(mobile,.7) 
# 30% of the dataset for validation purposes
x_test <- setdiff(mobile, x_train)

# We create the model based on classification  trees
arbol <- rpart(formula = price_range ~ ., data = x_train)

# To show the importance of the variables used in the model
sprintf("variable.importance = %s ",c(summary(arbol)$variable.importance))

# Displays the graph of the created tree
rpart.plot(arbol)

# Predict using the training set 
# (IMPORTANT NOTE: these are predictions for the TRAINING SET)
predict_train <- predict(arbol,x_train, type = "class")
# We obtain the metrics
cfm_train <- confusionMatrix(data =predict_train,
                             reference =x_train$price_range)

# Accuracy of the model (training set)
sprintf('Accuracy = %10.2f',cfm_train$overall[1]*100)

# Prediction using decision tree
prediccion <- predict(arbol, 
                      newdata = x_test, type = "class")

# We train our model
cfm_arbol <- confusionMatrix(prediccion,x_test[["price_range"]])
# Metrics of the model                             
cfm_arbol$overall

# We graph the confusion matrix
ggplot(data = as.data.frame(cfm_arbol$table),
       # Prediction vs actual values
       aes(x = Reference, y = Prediction)) + 
  geom_tile(aes(fill = log(Freq)), 
            colour = "white") +
  geom_text(aes(x = Reference, y = Prediction, 
                label = Freq),color="white") +
  labs(
    title =  'Confusion Matrix',
    subtitle = 'Predictions using the Test Set',
    x = "Actual Values",
    y = "Predictions"
  )+
  theme_minimal()+
  theme(
    title = element_text(size=14),
    axis.title=element_text(size=12, face="bold"),
    axis.text.x=element_text(size=12),
    axis.text.y=element_text(size=12),
    legend.position = "none"
  ) +
  scale_colour_gradient2()


# 3.2.2. Random Forest

# To create the predictive model
set.seed(2000)
# We create the random forest model
RF_model<-randomForest(price_range ~ ., data = x_train, importance=TRUE, ntree = 300)
RF_model

# Prediction with the test set
prediccion_RF <- predict(RF_model, newdata = x_test, type = "class")

# We create the confusion matrix (estimated price range vs actual values)
# Test set
cfm_RF <- confusionMatrix(data = prediccion_RF,
                          reference =x_test$price_range)
ggplot(data = as.data.frame(cfm_RF$table),
       aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = log(Freq)),
            colour = "white") +
  geom_text(aes(x = Reference, y = Prediction, 
                label = Freq),color="white") +
  labs(
    title =  'Confusion Matrix',
    subtitle = 'Predictions for the Test Set',
    x = "Actual Values",
    y = "Predictions"
  )+
  theme_minimal()+
  theme(
    title = element_text(size=14),
    axis.title=element_text(size=12, face="bold"),
    axis.text.x=element_text(size=12),
    axis.text.y=element_text(size=12),
    legend.position = "none"
  ) +
  scale_colour_gradient2()

# Precision of the RandomForest model (with the TEST SET)
sprintf('Accuracy = %10.2f',cfm_RF$overall[1]*100)


# 4. Results and Discussion

# We create a table with the statistics Accuracy/Kappa
acc<-data.frame(
  "Accuracy"= c(round((cfm_arbol$overall[1])*100,2),round((cfm_RF$overall[1])*100,2)),
  "Kappa" = c(round((cfm_RF$overall[2])*100,2),round((cfm_RF$overall[2])*100,2))
)

# Graph of the precision percentage of the two used models
qplot(c('Dec. Tree','Random Forest'), acc$Accuracy,main = 'Accuracy: Decision Tree vs Random Forest',
      ylab = 'Accuracy',xlab = 'Models Used',color = I("red"),size= I(2))

