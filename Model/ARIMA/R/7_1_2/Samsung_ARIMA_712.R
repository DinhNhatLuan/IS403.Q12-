# 1. CÀI VÀ LOAD PACKAGE ---------------------------------
rm(list = ls())
packages <- c(
  "readr", "dplyr", "lubridate", "ggplot2",
  "forecast", "scales", "Metrics",
  "tsibble", "fable", "fabletools", "feasts", "tidyr"
)

new_pkgs <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)

library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(forecast)
library(scales)
library(Metrics)
library(tsibble)
library(fable)
library(fabletools)
library(feasts)
library(tidyr)

# 2. ĐỌC DỮ LIỆU & TIỀN XỬ LÝ ----------------------------
# Thay đường dẫn file CSV bên dưới cho đúng
rm(list = ls())
data_path <- "C:/Users/capta/Downloads/IS403.Q12-/Data/Dataset/Samsung_clean.csv"

cat("\nĐọc dữ liệu từ:", data_path, "\n")
df <- read_csv(data_path)
cat("Số dòng:", nrow(df), "\n")

df <- df %>%
  mutate(Date = mdy(Date)) %>%
  arrange(Date)

glimpse(df)
head(df)

# 3. TẠO CHUỖI THỜI GIAN --------------------------------
ts_data <- df %>% select(Date, Close)
tsbl <- ts_data %>% as_tsibble(index = Date)

# 4. CHIA TẬP TRAIN / VALIDATE / TEST --------------------
n <- nrow(tsbl)
train_end <- floor(0.70 * n)
val_end   <- floor(0.80 * n)

train <- tsbl %>% slice(1:train_end)
validate <- tsbl %>% slice((train_end+1):val_end)
test <- tsbl %>% slice((val_end+1):n)

cat("Kích thước:\n")
cat(" Train:", nrow(train), "\n Validate:", nrow(validate), "\n Test:", nrow(test), "\n")

train_ts <- ts(train$Close, frequency = 1)
train_val_ts <- ts(c(train$Close, validate$Close), frequency = 1)
train_val_test_ts <- ts(c(train$Close, validate$Close, test$Close), frequency = 1)

# 5. HUẤN LUYỆN AUTO ARIMA -------------------------------
cat("\n--- Bắt đầu tìm mô hình Auto ARIMA ---\n")
fit <- auto.arima(
  train_ts,
  seasonal = FALSE,
  stepwise = TRUE,
  approximation = FALSE,
  ic = "aicc",
  method = "ML",
  allowdrift = TRUE,
  allowmean = TRUE,
  stationary = FALSE,
  trace = TRUE
)
cat("\n--- Kết quả mô hình ---\n")
print(summary(fit))

# 6. DỰ BÁO VALIDATE / TEST / TƯƠNG LAI -----------------
h_val <- nrow(validate)
fc_val <- forecast(fit, h = h_val)
y_pred_val <- as.numeric(fc_val$mean)

# Huấn luyện lại trên Train + Validate
fit_retrain <- Arima(
  train_val_ts,
  order = fit$arma[c(1,6,2)],
  include.drift = ("drift" %in% names(coef(fit))),
  include.mean = ("intercept" %in% names(coef(fit))),
  method = "ML"
)
cat("\n--- Mô hình retrain (Train+Validate) ---\n")
print(summary(fit_retrain))

# Dự báo trên test
h_test <- nrow(test)
fc_test <- forecast(fit_retrain, h = h_test)
y_pred_test <- as.numeric(fc_test$mean)

# Huấn luyện lại trên Train + Validate + Test trước dự đoán thực tế
fit_final <- Arima(
  train_val_test_ts,
  order = fit$arma[c(1,6,2)],
  include.drift = ("drift" %in% names(coef(fit))),
  include.mean = ("intercept" %in% names(coef(fit))),
  method = "ML"
)
cat("\n--- Mô hình retrain (Train+Validate+Test) ---\n")
print(summary(fit_final))

# Dự báo tương lai 30 ngày
h_future <- 30
fc_future <- forecast(fit_final, h = h_future)
future_dates <- seq(from = max(df$Date) + lubridate::days(1), by = "day", length.out = h_future)
forecast_df <- tibble(Date = future_dates, Predicted_Close = as.numeric(fc_future$mean))

cat("\n--- Dự đoán tương lai (30 ngày) ---\n")
print(forecast_df)

# 7. ĐÁNH GIÁ MÔ HÌNH -----------------------------------
msle <- function(actual, pred) {
  mean((log1p(pred) - log1p(actual))^2)
}

actual_val <- validate$Close
actual_test <- test$Close

# --- VALIDATE ---
rmse_val <- rmse(actual_val, y_pred_val)
mape_val <- mape(actual_val, y_pred_val) 
msle_val <- msle(actual_val, y_pred_val)

# --- TEST ---
rmse_test <- rmse(actual_test, y_pred_test)
mape_test <- mape(actual_test, y_pred_test) 
msle_test <- msle(actual_test, y_pred_test)

metrics <- tibble(
  Set = c("Validate", "Test"),
  RMSE = c(rmse_val, rmse_test),
  MAPE = c(mape_val, mape_test),
  MSLE = c(msle_val, msle_test)
)

cat("\n--- Kết quả đánh giá ---\n")
print(metrics, n = Inf)

# 8. VẼ BIỂU ĐỒ -----------------------------------------
# Chuẩn bị dữ liệu để vẽ
plot_df <- bind_rows(
  train %>% mutate(Set = "Train"),
  validate %>% mutate(Set = "Validate"),
  test %>% mutate(Set = "Test")
)

# Dự đoán (Validate, Test)
pred_val_df <- validate %>% mutate(Pred = y_pred_val)
pred_test_df <- test %>% mutate(Pred = y_pred_test)

# Tạo biểu đồ
p <- ggplot() +
  # Dữ liệu thực tế
  geom_line(data = train, aes(x = Date, y = Close, color = "Train"), size = 0.6) +
  geom_line(data = validate, aes(x = Date, y = Close, color = "Validate"), size = 0.6) +
  geom_line(data = test, aes(x = Date, y = Close, color = "Test"), size = 0.6) +
  
  # Dự đoán (Validation, Test, Future)
  geom_line(data = pred_val_df, aes(x = Date, y = Pred, linetype = "Predict (Validate)"), 
            color = "#8B4513", size = 0.8) +
  geom_line(data = pred_test_df, aes(x = Date, y = Pred, linetype = "Predict (Test)"), 
            color = "purple", size = 0.8) +
  geom_line(data = forecast_df, aes(x = Date, y = Predicted_Close, linetype = "Forecast Next 30 Days"), 
            color = "red", size = 0.8) +
  
  # Tiêu đề và trục
  labs(
    title = sprintf("Samsung Closing Price (%s → %s)", min(df$Date), forecast_df$Date[nrow(forecast_df)]),
    x = "Date", y = "Closing Price",
    color = "Dataset", linetype = ""
  ) +
  # Màu sắc và theme
  scale_color_manual(values = c("Train" = "blue", "Validate" = "orange", "Test" = "green")) +
  scale_linetype_manual(values = c("Predict (Validate)" = "dashed", 
                                   "Predict (Test)" = "dashed", 
                                   "Forecast Next 30 Days" = "dashed")) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
# Hiển thị nếu trong RStudio
if (interactive()) print(p)

# Lưu file hình
save_path <- file.path("C:/Users/capta/Downloads/IS403.Q12-/Model/ARIMA/R/7_1_2/Image/Samsung_ARIMA_712_R.png")
ggsave(filename = save_path, plot = p, width = 12, height = 6, dpi = 300)
cat(paste0("\n[OK] Biểu đồ đã lưu tại: ", save_path, "\n"))

