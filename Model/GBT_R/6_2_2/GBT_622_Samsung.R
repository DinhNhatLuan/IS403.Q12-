# --- CÀI & GỌI THƯ VIỆN ---

if(!require(gbm)) install.packages("gbm"); library(gbm)
if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(scales)) install.packages("scales"); library(scales)
if(!require(lubridate)) install.packages("lubridate"); library(lubridate)
if(!require(Metrics)) install.packages("Metrics"); library(Metrics)

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

# --- CHUYỂN DỮ LIỆU SANG DẠNG DATAFRAME ---

train_df <- as.data.frame(train_ds$X)
train_df$y <- train_ds$y

val_df <- as.data.frame(val_ds$X)
val_df$y <- val_ds$y

test_df <- as.data.frame(test_ds$X)
test_df$y <- test_ds$y

# --- HUẤN LUYỆN MÔ HÌNH GBT ---

set.seed(123)
gbt_model <- gbm(
  formula = y ~ .,
  data = train_df,
  distribution = "gaussian",
  n.trees = 5000,
  interaction.depth = 4,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  bag.fraction = 0.8,
  train.fraction = 1.0,
  cv.folds = 5,
  verbose = FALSE
)

best_iter <- gbm.perf(gbt_model, method = "cv")

# --- DỰ BÁO ---

pred_test_scaled <- predict(gbt_model, newdata = test_df, n.trees = best_iter)

# --- NGHỊCH CHUẨN HÓA ---

inverse_minmax <- function(x) x * (scale_max - scale_min) + scale_min
pred_test <- inverse_minmax(pred_test_scaled)
price_rescaled <- inverse_minmax(price_scaled)

# --- DỰ BÁO 30 NGÀY TỚI ---

future_days <- 30
last_seq <- tail(price_scaled, time_step)
future_scaled <- c()

for (i in 1:future_days) {
  x_input <- as.data.frame(t(last_seq))
  colnames(x_input) <- paste0("V", 1:time_step)
  yhat <- predict(gbt_model, newdata = x_input, n.trees = best_iter)
  future_scaled <- c(future_scaled, yhat)
  last_seq <- c(tail(last_seq, time_step - 1), yhat)
}

future_rescaled <- inverse_minmax(future_scaled)
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
    title = "Dự báo xu hướng giá cổ phiếu (GBT nối tiếp 30 ngày)",
    x = "Ngày", y = "Giá (USD)", color = ""
  ) +
  theme_minimal(base_size = 14)
