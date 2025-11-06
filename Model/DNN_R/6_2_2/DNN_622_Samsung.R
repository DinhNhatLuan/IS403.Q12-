# --- CÀI & GỌI THƯ VIỆN ---
if(!require(keras3)) install.packages("keras3"); library(keras3)
if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(scales)) install.packages("scales"); library(scales)
if(!require(lubridate)) install.packages("lubridate"); library(lubridate)

# --- ĐỌC DỮ LIỆU ---
data <- read.csv("C:/Users/admin/Documents/HK5 2025-2026/PTDLKD IS403.Q12/DoAn/Dataset/Samsung_clean.csv")

# --- XỬ LÝ NGÀY ---
date_col <- names(data)[grepl("date", tolower(names(data)))]
data$Date <- as.Date(data[[date_col[1]]])

# --- CHỌN CỘT CLOSE ---
price <- data$Close
scale_min <- min(price)
scale_max <- max(price)
price_scaled <- (price - scale_min) / (scale_max - scale_min)

# --- HÀM TẠO DỮ LIỆU CHUỖI ---
create_dataset <- function(series, time_step = 30) {
  X <- matrix(0, nrow = length(series) - time_step, ncol = time_step)
  y <- numeric(length(series) - time_step)
  for (i in 1:(length(series) - time_step)) {
    X[i,] <- series[i:(i + time_step - 1)]
    y[i] <- series[i + time_step]
  }
  list(X = X, y = y)
}

# --- CHIA DỮ LIỆU ---
n <- length(price)
train_end <- floor(0.6 * n)
val_end   <- floor(0.8 * n)
train <- price_scaled[1:train_end]
val   <- price_scaled[(train_end+1):val_end]
test  <- price_scaled[(val_end+1):n]

time_step <- 30
train_ds <- create_dataset(train, time_step)
val_ds   <- create_dataset(val, time_step)
test_ds  <- create_dataset(test, time_step)

# --- TẠO MÔ HÌNH DNN ---
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = c(time_step)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1)

model %>% compile(optimizer = "adam", loss = "mse", metrics = c("mae"))

# --- HUẤN LUYỆN ---
history <- model %>% fit(
  train_ds$X, train_ds$y,
  epochs = 100,
  batch_size = 32,
  validation_data = list(val_ds$X, val_ds$y),
  callbacks = list(callback_early_stopping(patience = 10, restore_best_weights = TRUE))
)

# --- DỰ BÁO ---
pred_test_scaled <- model %>% predict(test_ds$X)

# --- NGHỊCH CHUẨN HÓA ---
inverse_minmax <- function(x) x * (scale_max - scale_min) + scale_min
pred_test <- inverse_minmax(pred_test_scaled)
price_rescaled <- inverse_minmax(price_scaled)

# --- DỰ BÁO 30 NGÀY TỚI ---
future_days <- 30
last_seq <- tail(price_scaled, time_step)
future_scaled <- c()

for (i in 1:future_days) {
  x_input <- matrix(last_seq, nrow = 1, ncol = time_step)
  yhat <- model %>% predict(x_input)
  future_scaled <- c(future_scaled, yhat)
  last_seq <- c(tail(last_seq, time_step - 1), yhat)
}

future_rescaled <- inverse_minmax(future_scaled)

# --- NGÀY DỰ BÁO ---
future_dates <- seq(from = max(data$Date) + 1, by = "day", length.out = future_days)

# --- GHÉP DỮ LIỆU ---
df_actual <- tibble(Date = data$Date, Actual = price_rescaled)
df_future <- tibble(Date = future_dates, Forecast = future_rescaled)
df_last <- tail(df_actual, 1) %>% mutate(Forecast = Actual)

# --- VẼ BIỂU ĐỒ ---
ggplot() +
  geom_line(data = df_actual, aes(x = Date, y = Actual, color = "Thực tế"), linewidth = 1) +
  geom_line(
    data = bind_rows(df_last, df_future),
    aes(x = Date, y = Forecast, color = "Dự báo (30 ngày tới)"),
    linewidth = 1.2
  ) +
  scale_color_manual(values = c("Thực tế" = "blue", "Dự báo (30 ngày tới)" = "red")) +
  labs(
    title = "Dự báo xu hướng giá cổ phiếu (DNN nối tiếp 30 ngày)",
    x = "Ngày", y = "Giá (USD)", color = ""
  ) +
  theme_minimal(base_size = 14)