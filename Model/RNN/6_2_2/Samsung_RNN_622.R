# ==========================================================
# Samsung_RNN_622.R
# DỰ BÁO GIÁ ĐÓNG CỬA CỔ PHIẾU Samsung BẰNG MÔ HÌNH RNN
# (Phiên bản 622: Tỷ lệ 60-20-20, RNN 32 units, 30 epochs)
# ==========================================================

# 1. CÀI VÀ LOAD PACKAGE ---------------------------------
rm(list = ls())
packages <- c(
  "readr", "dplyr", "lubridate", "ggplot2",
  "scales", "Metrics", "keras", "tensorflow", "tibble"
)

new_pkgs <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)

library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(scales)
library(Metrics)
library(keras)
library(tensorflow)
library(tibble)

# 2. ĐỌC DỮ LIỆU & TIỀN XỬ LÝ ----------------------------
data_path <- "Samsung_clean.csv"
cat("\nĐọc dữ liệu từ:", data_path, "\n")

df <- read_csv(data_path)
cat("Số dòng:", nrow(df), "\n")

df <- df %>%
  mutate(Date = ymd(Date)) %>%
  arrange(Date)

glimpse(df)
head(df)

# 3. CHUẨN BỊ DỮ LIỆU RNN --------------------------------
data <- df$Close
n <- length(data)
train_end <- floor(0.60 * n)
val_end <- floor(0.80 * n)

train <- data[1:train_end]
validate <- data[(train_end + 1):val_end]
test <- data[(val_end + 1):n]

cat("Kích thước:\n")
cat(" Train:", length(train), "\n Validate:", length(validate), "\n Test:", length(test), "\n")

# Chuẩn hóa dữ liệu (MinMax)
min_val <- min(data)
max_val <- max(data)
scale_data <- function(x) (x - min_val) / (max_val - min_val)
inv_scale <- function(x) x * (max_val - min_val) + min_val

scaled_data <- scale_data(data)

# ==========================================================
# HÀM TẠO SEQUENCE (đảm bảo định dạng và ép kiểu float32)
# ==========================================================
create_sequences <- function(series, timesteps = 10) {
  n_samples <- length(series) - timesteps
  X_list <- vector("list", n_samples)
  Y_list <- vector("list", n_samples)
  
  for (i in seq_len(n_samples)) {
    X_list[[i]] <- as.numeric(series[i:(i + timesteps - 1)])
    Y_list[[i]] <- as.numeric(series[i + timesteps])
  }
  
  X_array <- array(unlist(X_list),
                   dim = c(as.integer(n_samples), as.integer(timesteps), as.integer(1)))
  Y_array <- array(unlist(Y_list),
                   dim = c(as.integer(n_samples), as.integer(1)))
  
  # Ép kiểu float32 cho TensorFlow
  X_array <- tf$cast(X_array, dtype = tf$float32)
  Y_array <- tf$cast(Y_array, dtype = tf$float32)
  
  list(X = X_array, Y = Y_array)
}

timesteps <- 10
train_seq <- create_sequences(scaled_data[1:train_end], timesteps)
val_seq <- create_sequences(scaled_data[(train_end + 1):val_end], timesteps)
test_seq <- create_sequences(scaled_data[(val_end + 1):n], timesteps)

# 4. XÂY DỰNG MÔ HÌNH RNN --------------------------------
cat("\n--- Khởi tạo mô hình RNN (622) ---\n")

model <- keras_model_sequential(list(
  layer_input(shape = c(as.integer(timesteps), 1L)),
  layer_simple_rnn(units = 32, activation = "tanh"),
  layer_dense(units = 1)
))

model$compile(
  optimizer = "adam",
  loss = "mse"
)

summary(model)

# 5. HUẤN LUYỆN MÔ HÌNH ----------------------------------
cat("\n--- Bắt đầu huấn luyện ---\n")

history <- model$fit(
  x = train_seq$X, y = train_seq$Y,
  validation_data = list(val_seq$X, val_seq$Y),
  epochs = as.integer(30),
  batch_size = as.integer(32),
  verbose = 1
)

# 6. DỰ BÁO & ĐÁNH GIÁ -----------------------------------
pred_val <- model$predict(val_seq$X)
pred_test <- model$predict(test_seq$X)

# Chuyển về giá trị gốc
pred_val_inv <- inv_scale(as.numeric(pred_val))
pred_test_inv <- inv_scale(as.numeric(pred_test))
actual_val_inv <- inv_scale(as.numeric(val_seq$Y))
actual_test_inv <- inv_scale(as.numeric(test_seq$Y))

# Hàm MSLE
msle <- function(actual, pred) {
  mean((log1p(pred) - log1p(actual))^2)
}

# --- VALIDATE ---
rmse_val <- rmse(actual_val_inv, pred_val_inv)
mape_val <- mape(actual_val_inv, pred_val_inv)
msle_val <- msle(actual_val_inv, pred_val_inv)

# --- TEST ---
rmse_test <- rmse(actual_test_inv, pred_test_inv)
mape_test <- mape(actual_test_inv, pred_test_inv)
msle_test <- msle(actual_test_inv, pred_test_inv)

metrics <- tibble(
  Set = c("Validate", "Test"),
  RMSE = c(rmse_val, rmse_test),
  MAPE = c(mape_val, mape_test),
  MSLE = c(msle_val, msle_test)
)

cat("\n--- Kết quả đánh giá ---\n")
print(metrics, n = Inf)

# 7. DỰ BÁO 30 NGÀY TƯƠNG LAI ----------------------------
cat("\n--- Dự báo 30 ngày tiếp theo ---\n")
last_sequence <- scaled_data[(length(scaled_data) - timesteps + 1):length(scaled_data)]
future_scaled <- c()

for (i in 1:30) {
  input_seq <- array(last_sequence, dim = c(1L, as.integer(timesteps), 1L))
  input_seq <- tf$cast(input_seq, dtype = tf$float32)
  
  next_val <- as.numeric(model$predict(input_seq))
  future_scaled <- c(future_scaled, next_val)
  last_sequence <- c(last_sequence[-1], next_val)
}

future_inv <- inv_scale(future_scaled)
future_dates <- seq(from = max(df$Date) + days(1), by = "day", length.out = 30)
forecast_df <- tibble(Date = future_dates, Predicted_Close = future_inv)

print(forecast_df)

# 8. VẼ BIỂU ĐỒ -----------------------------------------
p <- ggplot() +
  geom_line(data = df, aes(x = Date, y = Close, color = "Actual"), size = 0.6) +
  geom_line(data = tibble(Date = df$Date[(train_end + timesteps + 1):(val_end)],
                          Pred = pred_val_inv),
            aes(x = Date, y = Pred, linetype = "Predict (Validate)"), color = "#8B4513", size = 0.8) +
  geom_line(data = tibble(Date = df$Date[(val_end + timesteps + 1):n],
                          Pred = pred_test_inv),
            aes(x = Date, y = Pred, linetype = "Predict (Test)"), color = "purple", size = 0.8) +
  geom_line(data = forecast_df, aes(x = Date, y = Predicted_Close, linetype = "Forecast Next 30 Days"),
            color = "red", size = 0.8) +
  labs(
    title = sprintf("Samsung Closing Price Prediction (RNN 622, %s → %s)",
                    min(df$Date), forecast_df$Date[nrow(forecast_df)]),
    x = "Date", y = "Closing Price",
    color = "Dataset", linetype = ""
  ) +
  scale_color_manual(values = c("Actual" = "blue")) +
  scale_linetype_manual(values = c(
    "Predict (Validate)" = "dashed",
    "Predict (Test)" = "dashed",
    "Forecast Next 30 Days" = "dashed"
  )) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

save_path <- "Samsung_RNN_622_R.png"
ggsave(filename = save_path, plot = p, width = 12, height = 6, dpi = 300)
cat(paste0("\n[OK] Biểu đồ đã lưu tại: ", save_path, "\n"))
