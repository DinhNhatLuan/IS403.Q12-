library(depmixS4)
library(ggplot2)
library(zoo)

# Load data
CSV_PATH <- "Apple_clean.csv"  
df0 <- read.csv(CSV_PATH, check.names = FALSE)
names(df0) <- gsub("\\s+", "_", names(df0))

# Tìm cột ngày & close
date_col <- if ("Date" %in% names(df0)) "Date" else names(df0)[grep("date|time|timestamp", tolower(names(df0)))[1]]
close_cands <- c("Close","Adj_Close","AdjClose","Price","close","adj_close","price")
close_col <- close_cands[close_cands %in% names(df0)][1]
if (is.na(close_col)) close_col <- names(df0)[sapply(df0, is.numeric)][1]

df <- df0[, c(date_col, close_col)]
names(df) <- c("Date","Close")
df$Date  <- as.Date(df$Date)
df <- df[order(df$Date), ]

# Feature 1: LogReturn 
df$LogReturn <- c(NA, diff(log(df$Close)))
df <- na.omit(df) 

#  Split 70/10/20 
T <- nrow(df)
n_train <- floor(0.7*T); n_val <- floor(0.1*T)
d_train <- df[1:n_train, ]
d_val   <- df[(n_train+1):(n_train+n_val), ]
d_test  <- df[(n_train+n_val+1):T, ]

# Chuẩn hoá 1D (LogReturn)
log_mu <- mean(d_train$LogReturn); log_sd <- sd(d_train$LogReturn)


scale2 <- function(x, mu, sd) (x - mu)/sd
inv2   <- function(x, mu, sd)  x*sd + mu

d_train$lr_s <- scale2(d_train$LogReturn, log_mu, log_sd)
d_val$lr_s   <- scale2(d_val$LogReturn,   log_mu, log_sd)
d_test$lr_s  <- scale2(d_test$LogReturn,  log_mu, log_sd)


#  Fit HMM 
set.seed(42)
mod_712 <- depmix(
  response = list(lr_s ~ 1), 
  data = d_train,
  nstates = 7, 
  family = list(gaussian()) 
)
fit_712 <- fit(mod_712, verbose = FALSE)

#  Lấy mean LogReturn gốc cho từng state 
state_mu_lr_s  <- sapply(fit_712@response, function(st) st[[1]]@parameters$coefficients["(Intercept)"])
state_mu_lr    <- inv2(state_mu_lr_s, log_mu, log_sd)  

#  Posterior & dự báo đường giá 
post_train <- posterior(fit_712)  # gamma cho train
post_val   <- posterior(fit_712, newdata = d_val)
post_test  <- posterior(fit_712, newdata = d_test)

Eret_train <- as.matrix(post_train[, paste0("S",1:7)]) %*% matrix(state_mu_lr, ncol=1) # S1..S7
Eret_val   <- as.matrix(post_val[,   paste0("S",1:7)]) %*% matrix(state_mu_lr, ncol=1)
Eret_test  <- as.matrix(post_test[,  paste0("S",1:7)]) %*% matrix(state_mu_lr, ncol=1)

rebuild_prices <- function(p0, exp_rets) {
  p0 * exp(cumsum(exp_rets))
}
pred_train <- rebuild_prices(d_train$Close[1], Eret_train)
pred_val   <- rebuild_prices(d_train$Close[nrow(d_train)], Eret_val)
pred_test  <- rebuild_prices(d_val$Close[nrow(d_val)],     Eret_test)

#  Dự báo 30 ngày 
gamma_last <- post_test[nrow(post_test), paste0("S",1:7)]
state_dist <- gamma_last
pred30_returns <- c()
transmat <- fit_712@transition
for (i in 1:30) {
  pred30_returns[i] <- state_dist %*% state_mu_lr

  state_dist <- state_dist %*% as.matrix(transmat)[,,i] 
}
# Lấy giá đóng cửa cuối cùng của tập test làm giá khởi điểm
start_price_30d <- d_test$Close[nrow(d_test)] 
pred30_prices <- rebuild_prices(start_price_30d, pred30_returns)

# Tạo dãy ngày cho 30 ngày forecast
last_date_test <- d_test$Date[nrow(d_test)]
# Sử dụng seq.Date để tạo 30 ngày làm việc tiếp theo
dates30 <- seq.Date(from = last_date_test, length.out = 31, by = "day")
dates30 <- dates30[weekdays(dates30) %in% c("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday")][-1] # Loại bỏ ngày đầu tiên và giữ 30 ngày làm việc
if(length(dates30) > 30) dates30 <- dates30[1:30] # Đảm bảo chỉ 30 ngày

# (MAPE/RMSE/MSLE) 
mape <- function(a,p) mean(abs((a-p)/a))
rmse <- function(a,p) sqrt(mean((a-p)^2))
msle <- function(a,p) mean((log1p(a) - log1p(p))^2)

val_mape <- mape(d_val$Close,  pred_val)
val_rmse <- rmse(d_val$Close,  pred_val)
val_msle <- msle(d_val$Close,  pred_val)

test_mape <- mape(d_test$Close, pred_test)
test_rmse <- rmse(d_test$Close, pred_test)
test_msle <- msle(d_test$Close, pred_test)

cat(sprintf("Validation: MAPE=%.6f  RMSE=%.6f  MSLE=%.8f\n", val_mape, val_rmse, val_msle))
cat(sprintf("Test      : MAPE=%.6f  RMSE=%.6f  MSLE=%.8f\n", test_mape, test_rmse, test_msle))

#Plot 
p <- ggplot() +
  geom_line(aes(d_train$Date, d_train$Close, color="Train"), linewidth=0.7) +
  geom_line(aes(d_val$Date,   d_val$Close,   color="Validate"), linewidth=0.7) +
  geom_line(aes(d_test$Date,  d_test$Close,  color="Test"), linewidth=0.7) +
  geom_line(aes(d_val$Date,   pred_val,      color="Predict Validation"), linewidth=0.8, linetype="dashed") +
  geom_line(aes(d_test$Date,  pred_test,     color="Predict Test"), linewidth=0.8, linetype="dashed") +
  geom_line(aes(dates30,      pred30_prices, color="Predict30days"), linewidth=0.8, linetype="dashed") +
  scale_color_manual(NULL,
    breaks = c("Train","Validate","Predict Validation","Test","Predict Test", "Predict30days"),
    values = c("Train"="#1f77b4","Validate"="#ff7f0e","Predict Validation"="#2ca02c","Test"="#d62728","Predict Test"="#9467bd", "Predict30days"="saddlebrown")
  ) +
  labs(title="Apple Closing Price Data With Ratio 7_1_2", x="Date", y="Close value") +
  theme_minimal() + theme(legend.position="left")
print(p)