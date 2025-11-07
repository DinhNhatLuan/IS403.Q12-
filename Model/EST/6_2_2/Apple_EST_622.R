# --- Exponential Smoothing (ETS) - Apple_clean.csv ---
library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(forecast)
library(lubridate)
library(Metrics)

# 1. Đọc dữ liệu
data <- read_csv("Apple_clean.csv")

# 2. Chuẩn hóa cột Date
data$Date <- ymd(data$Date)
data <- data %>% arrange(Date)

# 3. Tạo tập train, validate, test (6:2:2)
n <- nrow(data)
train_size <- floor(0.6 * n)
val_size <- floor(0.2 * n)

train <- data[1:train_size, ]
val <- data[(train_size + 1):(train_size + val_size), ]
test <- data[(train_size + val_size + 1):n, ]

# 4. Tối ưu alpha
alphas <- seq(0.1, 0.9, by = 0.1)
best_alpha <- NULL
best_mape <- Inf
best_fit <- NULL
best_forecast_val <- NULL

for (a in alphas) {
  fit <- ses(train$Close, alpha = a, h = nrow(val))
  mape_val <- mape(val$Close, fit$mean)
  if (mape_val < best_mape) {
    best_mape <- mape_val
    best_alpha <- a
    best_fit <- fit
    best_forecast_val <- fit$mean
  }
}

cat("Best alpha:", best_alpha, "\n")

# 5. Kiểm tra trên test
final_fit <- ses(c(train$Close, val$Close), alpha = best_alpha, h = nrow(test))
forecast_test <- final_fit$mean

mape_test <- mape(test$Close, forecast_test)
rmse_test <- rmse(test$Close, forecast_test)
msle_test <- mean((log1p(test$Close) - log1p(forecast_test))^2)

cat("MAPE:", round(mape_test,4), "\nRMSE:", round(rmse_test,4), "\nMSLE:", round(msle_test,6), "\n")

# 6. Dự báo 30 ngày tiếp theo
last_date <- max(data$Date)
future_forecast <- ses(data$Close, alpha = best_alpha, h = 30)
future_values <- as.numeric(future_forecast$mean)
future_dates <- seq.Date(from = last_date, by = "day", length.out = 30)

# 7. Ghép toàn bộ dữ liệu để vẽ
plot_df <- data.frame(
  Date = c(train$Date, val$Date, test$Date, future_dates),
  Value = c(train$Close, val$Close, test$Close, rep(NA, 30)),
  PredictVal = c(rep(NA, nrow(train)), as.numeric(best_forecast_val), rep(NA, nrow(test) + 30)),
  PredictTest = c(rep(NA, nrow(train) + nrow(val)), as.numeric(forecast_test), rep(NA, 30)),
  PredictNext30 = c(rep(NA, nrow(train) + nrow(val) + nrow(test)), future_values)
)

# 8. Vẽ biểu đồ
p <- ggplot(plot_df, aes(x = Date)) +
  geom_line(aes(y = Value, color = "Actual"), linewidth = 1) +
  geom_line(aes(y = PredictVal, color = "Predict Validate"), linewidth = 1, linetype = "dashed") +
  geom_line(aes(y = PredictTest, color = "Predict Test"), linewidth = 1, linetype = "dotted") +
  geom_line(aes(y = PredictNext30, color = "Next 30 Days"), linewidth = 1, linetype = "dotdash") +
  scale_color_manual(values = c(
    "Actual" = "#1E3A8A",
    "Predict Validate" = "#3B82F6",
    "Predict Test" = "#F59E0B",
    "Next 30 Days" = "#10B981"
  )) +
  labs(
    title = "Apple - Exponential Smoothing (6:2:2)",
    subtitle = paste("Best α =", best_alpha),
    x = "Date",
    y = "Close Price",
    color = "Legend"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("Apple_EST_622.png", p, width = 10, height = 5, dpi = 300)
