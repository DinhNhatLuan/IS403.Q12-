# --- CÀI VÀ GỌI THƯ VIỆN ---
if(!require(keras)) install.packages("keras"); library(keras)
if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(scales)) install.packages("scales"); library(scales)

# --- ĐỌC DỮ LIỆU ---
data <- read.csv("D:/HK5_2025-2026/IS403-PTDLKD/DoAn/Samsung_clean.csv")

# --- XỬ LÝ NGÀY ---
date_col <- names(data)[grepl("date", tolower(names(data)))]
if (length(date_col) == 0) stop("Không tìm thấy cột ngày trong dữ liệu!")
date_col <- date_col[1]

data[[date_col]] <- as.character(data[[date_col]])

possible_formats <- c("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d")

convert_date_auto <- function(x) {
  for (fmt in possible_formats) {
    parsed <- suppressWarnings(as.Date(x, format = fmt))
    if (sum(!is.na(parsed)) > length(parsed) / 2) return(parsed)
  }
  suppressWarnings(as.Date(parse_date_time(x, orders = c("ymd", "dmy", "mdy"))))
}

data$Date <- convert_date_auto(data[[date_col]])

# --- CHỌN CỘT CLOSE ---
price <- data$Close
n <- length(price)

# --- CHUẨN HÓA DỮ LIỆU ---
scale_min <- min(price)
scale_max <- max(price)
price_scaled <- (price - scale_min) / (scale_max - scale_min)

# --- HÀM TẠO DỮ LIỆU CHUỖI THỜI GIAN ---
create_dataset <- function(series, time_step = 30) {
  X <- array(0, dim = c(length(series) - time_step, time_step, 1))
  y <- array(0, dim = c(length(series) - time_step))
  for (i in 1:(length(series) - time_step)) {
    X[i,,] <- series[i:(i + time_step - 1)]
    y[i] <- series[i + time_step]
  }
  list(X = X, y = y)
}

# --- CHIA DỮ LIỆU 7/1/2 ---
train_end <- floor(0.7 * n)
val_end   <- floor(0.8 * n)
train <- price_scaled[1:train_end]
val   <- price_scaled[(train_end+1):val_end]
test  <- price_scaled[(val_end+1):n]

# --- TẠO DỮ LIỆU CHO GRU ---
time_step <- 30
train_ds <- create_dataset(train, time_step)
val_ds   <- create_dataset(val, time_step)
test_ds  <- create_dataset(test, time_step)

# --- TẠO MÔ HÌNH GRU ---
model <- keras_model_sequential() %>%
  layer_gru(units = 64, input_shape = c(time_step, 1), return_sequences = FALSE) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = "adam",
  loss = "mse",
  metrics = c("mae")
)

# --- HUẤN LUYỆN ---
history <- model %>% fit(
  train_ds$X, train_ds$y,
  epochs = 50,
  batch_size = 32,
  validation_data = list(val_ds$X, val_ds$y),
  verbose = 1
)

# --- DỰ BÁO CHO TẬP TEST ---
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
  x_input <- array(last_seq, dim = c(1, time_step, 1))
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

# --- VẼ BIỂU ĐỒ ---
ggplot() +
  geom_line(data = df_actual, aes(x = Date, y = Actual, color = "Thực tế"), size = 1) +
  geom_line(
    data = bind_rows(tail(df_actual, 1), df_future),
    aes(x = Date, y = Forecast, color = "Dự báo (30 ngày tới)"),
    size = 1.2
  ) +
  scale_color_manual(values = c("Thực tế" = "blue", "Dự báo (30 ngày tới)" = "red")) +
  labs(
    title = "Dự báo xu hướng giá cổ phiếu (GRU nối tiếp 30 ngày)",
    x = "Ngày", y = "Giá (USD)", color = ""
  ) +
  theme_minimal(base_size = 14)
