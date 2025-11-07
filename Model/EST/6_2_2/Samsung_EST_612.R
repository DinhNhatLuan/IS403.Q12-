# ==============================================================
#  Samsung_clean.csv — Exponential Smoothing (ETS) (phần 6.2.2)
#  Logic giữ nguyên cấu trúc từ ARIMA_622, thay bằng ETS
#  Hoàn toàn dùng ymd(), ẩn warning, xử lý ngày an toàn
# ==============================================================

# ---- 1. Xóa môi trường, cài gói ----
rm(list = ls())
suppressMessages({
  library(readr)
  library(dplyr)
  library(lubridate)
  library(forecast)
  library(ggplot2)
  library(Metrics)
})

# ---- 2. Đọc dữ liệu ----
data <- read_csv("Samsung_clean.csv", show_col_types = FALSE)
data <- data %>%
  mutate(Date = ymd(Date)) %>%          # luôn dùng ymd()
  arrange(Date)

# kiểm tra cột
if (!"Date" %in% names(data) || !"Close" %in% names(data)) {
  stop("Dataset phải có cột Date và Close.")
}

# ---- 3. Chia train/validate/test (60/20/20) ----
n <- nrow(data)
train_size <- floor(0.6 * n)
val_size   <- floor(0.2 * n)

train_data <- data[1:train_size, ]
validate_data <- data[(train_size + 1):(train_size + val_size), ]
test_data <- data[(train_size + val_size + 1):n, ]

# ---- 4. Tạo chuỗi thời gian ----
train_ts <- ts(train_data$Close)
train_val_ts <- ts(c(train_data$Close, validate_data$Close))
full_ts <- ts(data$Close)

# ---- 5. Huấn luyện mô hình ETS trên tập train ----
suppressWarnings({
  ets_model <- ets(train_ts)
})

# ---- 6. Dự báo tập validate ----
future_val <- forecast(ets_model, h = length(validate_data$Close))
validate_data$Forecast <- as.numeric(future_val$mean)

# ---- 7. Đánh giá validate ----
mape_val <- mape(validate_data$Close, validate_data$Forecast)
rmse_val <- rmse(validate_data$Close, validate_data$Forecast)
msle_val <- mean((log1p(validate_data$Forecast) - log1p(validate_data$Close))^2)

cat("Validation metrics:\n")
cat("MAPE:", round(mape_val, 4), "\nRMSE:", round(rmse_val, 4), "\nMSLE:", round(msle_val, 6), "\n")

# ---- 8. Retrain trên train + validate ----
suppressWarnings({
  ets_model_tv <- ets(train_val_ts)
})
future_test <- forecast(ets_model_tv, h = length(test_data$Close))
test_data$Forecast <- as.numeric(future_test$mean)

# ---- 9. Đánh giá test ----
mape_test <- mape(test_data$Close, test_data$Forecast)
rmse_test <- rmse(test_data$Close, test_data$Forecast)
msle_test <- mean((log1p(test_data$Forecast) - log1p(test_data$Close))^2)

cat("\nTest metrics:\n")
cat("MAPE:", round(mape_test, 4), "\nRMSE:", round(rmse_test, 4), "\nMSLE:", round(msle_test, 6), "\n")

# ---- 10. Retrain toàn bộ + Dự báo 30 ngày tiếp theo (đã sửa lỗi seq.int) ----
suppressWarnings({
  ets_model_full <- ets(full_ts)
  future_forecast <- forecast(ets_model_full, h = 30)
})

# --- Xử lý ngày an toàn ---
suppressWarnings({
  data$Date_orig <- data$Date
  parsed_dates <- parse_date_time(data$Date_orig,
                                  orders = c("ymd", "Ymd", "dmy", "mdy", "Y-m-d", "m/d/Y", "d/m/Y"),
                                  exact = FALSE)
  data$Date_parsed <- as.Date(parsed_dates)
  
  if (all(is.na(data$Date_parsed))) {
    tmp <- gsub("\\.", "-", data$Date_orig)
    tmp <- gsub("/", "-", tmp)
    parsed2 <- parse_date_time(tmp, orders = c("ymd", "dmy", "mdy", "Y-m-d"))
    data$Date_parsed <- as.Date(parsed2)
  }
  
  valid_idx <- !is.na(data$Date_parsed)
  if (!any(valid_idx)) {
    warning("Không tìm thấy giá trị Date hợp lệ, dùng Sys.Date() làm fallback.")
    last_date <- Sys.Date()
  } else {
    data <- data[valid_idx, , drop = FALSE]
    last_date <- max(data$Date_parsed, na.rm = TRUE)
  }
  
  if (!is.finite(last_date) || is.na(last_date)) {
    warning("last_date không hợp lệ, dùng Sys.Date() fallback.")
    last_date <- Sys.Date()
  }
  
  future_h <- 30L
  future_dates <- seq.Date(from = last_date + 1, by = "day", length.out = future_h)
  pred_vals <- as.numeric(future_forecast$mean)
  if (length(pred_vals) != future_h) {
    warning(sprintf("Độ dài vector dự báo (%d) khác với future_h (%d).", length(pred_vals), future_h))
    if (length(pred_vals) < future_h) {
      pred_vals <- c(pred_vals, rep(NA_real_, future_h - length(pred_vals)))
    } else {
      pred_vals <- pred_vals[1:future_h]
    }
  }
  future_df <- data.frame(Date = future_dates, Forecast = pred_vals)
})

cat("\nDự báo tương lai từ", as.character(future_df$Date[1]), "đến", as.character(tail(future_df$Date, 1)), "\n")

# ---- 11. Vẽ biểu đồ ----
p <- ggplot() +
  geom_line(data = train_data, aes(x = Date, y = Close), color = "blue", linewidth = 0.8) +
  geom_line(data = validate_data, aes(x = Date, y = Close), color = "green", linewidth = 0.8) +
  geom_line(data = test_data, aes(x = Date, y = Close), color = "black", linewidth = 0.8) +
  geom_line(data = validate_data, aes(x = Date, y = Forecast), color = "orange", linewidth = 0.8, linetype = "dashed") +
  geom_line(data = test_data, aes(x = Date, y = Forecast), color = "red", linewidth = 0.8, linetype = "dashed") +
  geom_line(data = future_df, aes(x = Date, y = Forecast), color = "purple", linewidth = 0.9, linetype = "dotdash") +
  labs(title = "Samsung — Exponential Smoothing (ETS) Forecast 6.2.2",
       x = "Date", y = "Close Value") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

print(p)
ggsave("Samsung_EST_622.png", plot = p, width = 10, height = 5, dpi = 300)

# ---- 12. Kết thúc ----
cat("\n✅ Đã hoàn tất dự báo và lưu biểu đồ Samsung_EST_622.png\n")
