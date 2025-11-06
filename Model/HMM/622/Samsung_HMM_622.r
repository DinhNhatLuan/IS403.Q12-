library(depmixS4)
library(ggplot2)
library(zoo)

# Load data
CSV_PATH <- "Samsung_clean.csv"  
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

# Feature 1: LogReturn; Feature 2: Volatility(20)
df$LogReturn <- c(NA, diff(log(df$Close)))
df$Volatility <- rollapply(df$LogReturn, 20, sd, fill = NA, align = "right") * sqrt(252)
df <- na.omit(df)

# Split 60/20/20 
T <- nrow(df)
n_train <- floor(0.6*T); n_val <- floor(0.2*T)
d_train <- df[1:n_train, ]
d_val   <- df[(n_train+1):(n_train+n_val), ]
d_test  <- df[(n_train+n_val+1):T, ]

# Chuẩn hoá 2D 
log_mu <- mean(d_train$LogReturn); log_sd <- sd(d_train$LogReturn)
vol_mu <- mean(d_train$Volatility); vol_sd <- sd(d_train$Volatility)

scale2 <- function(x, mu, sd) (x - mu)/sd
inv2   <- function(x, mu, sd)  x*sd + mu

d_train$lr_s <- scale2(d_train$LogReturn, log_mu, log_sd)
d_train$vol_s<- scale2(d_train$Volatility, vol_mu, vol_sd)
d_val$lr_s   <- scale2(d_val$LogReturn,   log_mu, log_sd)
d_val$vol_s  <- scale2(d_val$Volatility,  vol_mu, vol_sd)
d_test$lr_s  <- scale2(d_test$LogReturn,  log_mu, log_sd)
d_test$vol_s <- scale2(d_test$Volatility, vol_mu, vol_sd)

#  Fit HMM 6 
set.seed(42)
mod_622 <- depmix(
  response = list(lr_s ~ 1, vol_s ~ 1),
  data = d_train,
  nstates = 6,
  family = list(gaussian(), gaussian())
)
fit_622 <- fit(mod_622, verbose = FALSE)

#  Lấy mean LogReturn gốc cho từng state 

state_mu_lr_s  <- sapply(fit_622@response, function(st) st[[1]]@parameters$coefficients["(Intercept)"])
state_mu_lr    <- inv2(state_mu_lr_s, log_mu, log_sd)  

#  dự báo đường giá
post_train <- posterior(fit_622)  

post_val   <- posterior(fit_622, newdata = d_val)
post_test  <- posterior(fit_622, newdata = d_test)

Eret_train <- as.matrix(post_train[, paste0("S",1:6)]) %*% matrix(state_mu_lr, ncol=1)
Eret_val   <- as.matrix(post_val[,   paste0("S",1:6)]) %*% matrix(state_mu_lr, ncol=1)
Eret_test  <- as.matrix(post_test[,  paste0("S",1:6)]) %*% matrix(state_mu_lr, ncol=1)

rebuild_prices <- function(p0, exp_rets) {
  p0 * exp(cumsum(exp_rets))
}
pred_train <- rebuild_prices(d_train$Close[1], Eret_train)
pred_val   <- rebuild_prices(d_train$Close[nrow(d_train)], Eret_val)
pred_test  <- rebuild_prices(d_val$Close[nrow(d_val)],     Eret_test)

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

# Plot
p <- ggplot() +
  geom_line(aes(d_train$Date, d_train$Close, color="Train"), linewidth=0.7) +
  geom_line(aes(d_val$Date,   d_val$Close,   color="Validate"), linewidth=0.7) +
  geom_line(aes(d_test$Date,  d_test$Close,  color="Test"), linewidth=0.7) +
  geom_line(aes(d_val$Date,   pred_val,      color="Predict Validation"), linewidth=0.8, linetype="dashed") +
  geom_line(aes(d_test$Date,  pred_test,     color="Predict Test"), linewidth=0.8, linetype="dashed") +
  scale_color_manual(NULL,
    breaks = c("Train","Validate","Predict Validation","Test","Predict Test"),
    values = c("Train"="#1f77b4","Validate"="#ff7f0e","Predict Validation"="#2ca02c","Test"="#d62728","Predict Test"="#9467bd")
  ) +
  labs(title="Samsung Closing Price Data With Ratio 6_2_2", x="Date", y="Close value") +
  theme_minimal() + theme(legend.position="left")
print(p)