# Install packages. Uncomment if needed
# install.packages("randomForest")
# install.packages("dplyr")
# install.packages("caret")
# install.packages("ggplot2")
# install.packages("ROCR")
# install.packages("reshape2")

# Load necessary library
library(randomForest)
library(dplyr)
library(caret)
library(ggplot2)
library(ROCR)
library(reshape2)

# Set seed for reproducibility
set.seed(001)

# Load dataset
creditrisk_data <- read.csv("INSERT_PATH")

# View the first few rows of the dataset
head(creditrisk_data)

# Summary of dataset
summary(creditrisk_data)

# Check for missing values. Displays amount of missing values in dataset
sum(is.na(creditrisk_data))

# Extract numeric columns from "creditrisk_data"
numeric_data <- creditrisk_data[, sapply(creditrisk_data, is.numeric)]

# Calculate the correlation matrix for all numeric columns
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Correlations with the "DEFAULT" column
cor_with_default <- cor_matrix[, "DEFAULT"]

# Convert the correlation results into a data frame without duplicating variable names
cor_df <- data.frame(Variable = rownames(as.data.frame(cor_with_default)), 
                     Correlation = cor_with_default, row.names = NULL)

# View the data frame to check for correctness
print(cor_df)

# Sort the data frame by correlation strength
cor_df <- cor_df[order(-cor_df$Correlation), ]

# View the sorted data frame
print(cor_df)

# Plot the correlation results using ggplot2
ggplot(cor_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +  # Flip coordinates for better readability
  labs(title = "Correlation of Variables with DEFAULT",
       x = "Variables",
       y = "Correlation") +
  theme_minimal()

# Boxplot for all numeric variables in one plot
boxplot(creditrisk_data[, numeric_cols], main="Boxplots for All Numeric Variables", 
        las=2,  # Rotate labels on x-axis for better readability
        col="blue")

# Print the AMOUNT variable in descending order since amount may contain outliers
print(sort(creditrisk_data$AMOUNT, decreasing = TRUE))

# Convert the "DEFAULT" column to a factor
creditrisk_data$DEFAULT <- as.factor(creditrisk_data$DEFAULT)

# Verify the conversion and check levels
levels(creditrisk_data$DEFAULT)

# Split the data into training (70%) and testing (30%) sets
train_index <- createDataPartition(creditrisk_data$DEFAULT, p = 0.7, list = FALSE)
train_data <- creditrisk_data[train_index, ]
test_data <- creditrisk_data[-train_index, ]

# Exclude specific columns using select() from dplyr
excluded_columns <- c("OBS#", "JOB", "PRESENT_RESIDENT", "MALE_DIV", "RETRAINING", "MALE_MAR_or_WID", "NUM_DEPENDENTS")
train_data <- train_data[, !(colnames(train_data) %in% excluded_columns)]
test_data <- test_data[, !(colnames(test_data) %in% excluded_columns)]

# Verify columns were excluded
colnames(train_data)
colnames(test_data)

# Train the Random Forest model
rf_model <- randomForest(DEFAULT ~ ., data = train_data, ntree = 1000, mtry = 4, nodesize = 5, importance = TRUE)

# Print the model summary to check the results
print(rf_model)

# Make predictions on the test set
test_predictions <- predict(rf_model, test_data)

# Ensure the predictions are factors with the same levels as the actual data
test_predictions <- factor(test_predictions, levels = levels(test_data$DEFAULT))

# Create the confusion matrix to evaluate performance
confusion_matrix <- confusionMatrix(test_predictions, test_data$DEFAULT)

# Print the confusion matrix and related metrics
print(confusion_matrix)

# View the importance of each variable
importance(rf_model)

# Plot the importance of the variables
varImpPlot(rf_model)

# Retrieve full metrics, including F1, if confusion matrix calculation supports it
f1_scores <- 2 * (confusion_matrix$byClass['Pos Pred Value'] * confusion_matrix$byClass['Sensitivity']) /
  (confusion_matrix$byClass['Pos Pred Value'] + confusion_matrix$byClass['Sensitivity'])

# Print F1-Scores for all classes
print(f1_scores)

test_probabilities <- predict(rf_model, test_data, type = "prob")

# Generate ROCR prediction object
pred <- prediction(test_probabilities[, 2], test_data$DEFAULT)

# Calculate performance metrics
perf <- performance(pred, "tpr", "fpr")  # True positive rate vs false positive rate (ROC curve)

# Plot ROC curve
plot(perf, col = "blue", main = "ROC Curve for Random Forest Model")

# Calculate AUC
auc <- performance(pred, measure = "auc")
auc_value <- auc@y.values[[1]]
print(auc_value)


# Verify Model using verification dataset

# Load the dataset for verification
verify_data <- read.csv("INSERT_PATH")

# Preprocess the verification dataset
verify_data <- verify_data[, !(colnames(verify_data) %in% excluded_columns)]
for (col in colnames(verify_data)) {
  if (is.factor(verify_data[[col]]) && col %in% colnames(train_data)) {
    verify_data[[col]] <- factor(verify_data[[col]], levels = levels(train_data[[col]]))
  }
}

# Make predictions with verification data
verify_predictions <- predict(rf_model, verify_data)

# Evaluate performance if actual outcomes are available
if ("DEFAULT" %in% colnames(verify_data)) {
  verify_data$DEFAULT <- as.factor(verify_data$DEFAULT)
  verify_confusion_matrix <- confusionMatrix(verify_predictions, verify_data$DEFAULT)
  print(verify_confusion_matrix)
} else {
  # Save predictions if no actual outcomes
  write.csv(data.frame(Predictions = verify_predictions), "verify_dataset_predictions.csv")
  print("Predictions saved to 'verify_dataset_predictions.csv'.")
}

# Metrics for testing and verification datasets
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1-Score"),
  Testing = c(
    confusion_matrix$overall['Accuracy'],
    confusion_matrix$byClass['Sensitivity'],
    confusion_matrix$byClass['Specificity'],
    2 * (confusion_matrix$byClass['Pos Pred Value'] * confusion_matrix$byClass['Sensitivity']) /
      (confusion_matrix$byClass['Pos Pred Value'] + confusion_matrix$byClass['Sensitivity'])
  ),
  Verification = if ("DEFAULT" %in% colnames(verify_data)) {
    c(
      verify_confusion_matrix$overall['Accuracy'],
      verify_confusion_matrix$byClass['Sensitivity'],
      verify_confusion_matrix$byClass['Specificity'],
      2 * (verify_confusion_matrix$byClass['Pos Pred Value'] * verify_confusion_matrix$byClass['Sensitivity']) /
        (verify_confusion_matrix$byClass['Pos Pred Value'] + verify_confusion_matrix$byClass['Sensitivity'])
    )
  } else {
    c(NA, NA, NA, NA)  # Use NA if the 'DEFAULT' column is missing
  }
)

# Reshape the data for ggplot
metrics_melted <- melt(metrics_df, id.vars = "Metric", variable.name = "Dataset", value.name = "Value")

# Reorder the "Metric" column explicitly
metrics_melted$Metric <- factor(metrics_melted$Metric, levels = c("Accuracy", "Sensitivity", "Specificity", "F1-Score"))

# Create the bar chart
ggplot(metrics_melted, aes(x = Metric, y = Value, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Value, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3.5) +
  labs(title = "Model Performance Comparison",
       y = "Value",
       x = "Metric") +
  scale_y_continuous(limits = c(0, 1)) +
  scale_fill_manual(values = c("Testing" = "darkgreen", "Verification" = "deepskyblue")) +
  theme_minimal()
