# ==============================
# Import Libraries
# ==============================
library(ggplot2)
library(scales)
library(Metrics)
library(dplyr)
library(lubridate)

# ==============================
# Read data
# ==============================
df <- read.csv("Samsung_clean.csv")
df$Date <- as.Date(df$Date)
df <- df %>% arrange(Date)
df_Close <- df["Close"]

# ==============================
# Data normalization
# ==============================
min_val <- min(df_Close$Close)
max_val <- max(df_Close$Close)
data_scaled <- (df_Close$Close - min_val) / (max_val - min_val)

# ==============================
# Data splitting → 70-10-20
# ==============================
n <- length(data_scaled)
train_size <- floor(0.7 * n)      # ← 70%
val_size   <- floor(0.1 * n)      # ← 10%
train_data <- data_scaled[1:train_size]
val_data   <- data_scaled[(train_size + 1):(train_size + val_size)]
test_data  <- data_scaled[(train_size + val_size + 1):n]

# ==============================
# Model training
# ==============================
x_train <- data.frame(x_train = 1:train_size)
y_train <- train_data
model <- lm(y_train ~ x_train, data = x_train)
summary(model)
intercept <- coef(model)[1]
coef_val <- coef(model)[2]
r_squared <- summary(model)$r.squared
cat("Intercept:", intercept, "\n")
cat("Coefficients:", coef_val, "\n")
cat("R-squared:", r_squared, "\n")

# ==============================
# Validation
# ==============================
x_val <- data.frame(x_train = (train_size + 1):(train_size + val_size))
y_val <- val_data
y_pred_val <- predict(model, newdata = x_val)
# Denormalize
y_val_inv <- y_val * (max_val - min_val) + min_val
y_pred_val_inv <- y_pred_val * (max_val - min_val) + min_val
val_mape <- mape(y_val_inv, y_pred_val_inv)
val_rmse <- rmse(y_val_inv, y_pred_val_inv)
val_msle <- msle(y_val_inv, y_pred_val_inv)
cat("MAPE (Validation):", val_mape, "\n")
cat("RMSE (Validation):", val_rmse, "\n")
cat("MSLE (Validation):", val_msle, "\n")

# ==============================
# Test
# ==============================
x_test <- data.frame(x_train = (train_size + val_size + 1):n)
y_test <- test_data
y_pred_test <- predict(model, newdata = x_test)
# Denormalize
y_test_inv <- y_test * (max_val - min_val) + min_val
y_pred_test_inv <- y_pred_test * (max_val - min_val) + min_val
test_mape <- mape(y_test_inv, y_pred_test_inv)
test_rmse <- rmse(y_test_inv, y_pred_test_inv)
test_msle <- msle(y_test_inv, y_pred_test_inv)
cat("MAPE (Test):", test_mape, "\n")
cat("RMSE (Test):", test_rmse, "\n")
cat("MSLE (Test):", test_msle, "\n")

# ==============================
# Predict next 30 days
# ==============================
last_index <- n
x_next_30_days <- data.frame(x_train = (last_index + 1):(last_index + 30))
y_next_30_days <- predict(model, newdata = x_next_30_days)
y_next_30_days_inv <- y_next_30_days * (max_val - min_val) + min_val
cat("Predicted closing prices for the next 30 days:\n")
print(y_next_30_days_inv)

# ==============================
# Generate future dates
# ==============================
last_date <- tail(df$Date, 1)
index_next_30_days <- seq(from = last_date + 1, by = "day", length.out = 30)

# ==============================
# Visualization
# ==============================
train_dates <- df$Date[1:train_size]
val_dates   <- df$Date[(train_size + 1):(train_size + val_size)]
test_dates  <- df$Date[(train_size + val_size + 1):n]

plot_data <- data.frame(
  Date = c(train_dates, val_dates, val_dates, test_dates, test_dates, index_next_30_days),
  Close = c(
    y_train * (max_val - min_val) + min_val,
    y_val_inv,
    y_pred_val_inv,
    y_test_inv,
    y_pred_test_inv,
    y_next_30_days_inv
  ),
  Type = c(
    rep("Train", length(y_train)),
    rep("Validate", length(y_val)),
    rep("PredictValidate", length(y_pred_val)),
    rep("Test", length(y_test)),
    rep("PredictTest", length(y_pred_test)),
    rep("Next30Days", length(y_next_30_days_inv))
  )
)

ggplot(plot_data, aes(x = Date, y = Close, color = Type)) +
  geom_line(size = 1) +
  theme_minimal() +
  labs(
    title = paste("Apple Closing Price (7_1_2) from",      # ← 7_1_2
                  format(df$Date[1], "%Y-%m-%d"), "to",
                  format(tail(index_next_30_days, 1), "%Y-%m-%d")),
    x = "Date",
    y = "Close value"
  ) +
  scale_color_manual(values = c(
    "Train" = "blue",
    "Validate" = "orange",
    "PredictValidate" = "red",
    "Test" = "green",
    "PredictTest" = "purple",
    "Next30Days" = "black"
  )) +
  theme(legend.position = "bottom")

ggsave("LR_712_Samsung.png", width = 14, height = 5)   
