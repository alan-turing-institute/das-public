#########################
# MODEL FITTING for DAS #
#########################
# Author: Giovanni Colavizza

require(ggplot2)

# load the dataset and make transformations
df <- read.csv("dataset/export_full.csv", sep = ";")
df$has_das <- factor(df$has_das)
df$is_plos <- factor(df$is_plos)
df$is_bmc <- factor(df$is_bmc)
df$has_month <- factor(df$has_month)
df$das_class <- factor(as.integer(df$das_class))
df$j_lower <- factor(df$j_lower)
df$journal_domain <- factor(df$journal_domain)
df$journal_field <- factor(df$journal_field)
df$journal_subfield <- factor(df$journal_subfield)

# filter for NaN and by time (assuming to use 3y citation window: exclude 2016 to 2018)
df_filtered <- df[(!is.na(df$h_index_mean))&(df$p_year<2016),]

# log-transform (add 1 to bound between zero and infinity)
df_filtered$n_cit_2_log <- df_filtered$n_cit_2 + 1
df_filtered$n_cit_2_log <- sapply(df_filtered$n_cit_2_log,log)
df_filtered$n_cit_3_log <- df_filtered$n_cit_3 + 1
df_filtered$n_cit_3_log <- sapply(df_filtered$n_cit_3_log,log)
df_filtered$n_cit_5_log <- df_filtered$n_cit_5 + 1
df_filtered$n_cit_5_log <- sapply(df_filtered$n_cit_5_log,log)

# log-transform of other variables (optional, but better fitting due to outliers)
df_filtered$n_authors <- sapply(df_filtered$n_authors,log)
df_filtered$h_index_mean <- df_filtered$h_index_mean + 1
df_filtered$h_index_mean <- sapply(df_filtered$h_index_mean,log)
df_filtered$n_references_tot <- df_filtered$n_references_tot + 1
df_filtered$n_references_tot <- sapply(df_filtered$n_references_tot,log)

# create the dataset only with cited papers and the boolean variable if a paper is cited
df_filtered_negative <- df_filtered[df_filtered$n_cit_3 > 0,]
df_filtered$is_cited <- factor(df_filtered$n_cit_3 > 0)

set.seed(101) # Set Seed so that same sample can be reproduced
# Now Selecting 75% of data as train and 25% as test
sample <- sample.int(n = nrow(df_filtered), size = floor(.75*nrow(df_filtered)), replace = F)
train <- df_filtered[sample, ]
test  <- df_filtered[-sample, ]

# select the dataset which will be used in regressions
DATASET <- df_filtered

# exploratory stats and plots (cf. Tables 4 and 5 in the paper)
summary(DATASET)
corr <- round(cor(DATASET[, c("n_cit_3_log", "n_authors", "p_year", "p_month", "h_index_mean", "h_index_median", "n_references_tot")], method = "pearson"), 2)
upper <- corr
upper[upper.tri(corr, diag = TRUE)] <- ""
upper <- as.data.frame(upper)
upper
ggpairs(DATASET[, c("n_cit_3_log", "n_authors", "p_year", "h_index_mean", "h_index_median", "n_references_tot")])

# DAS class frequencies
par(mfcol = c(2, 2))
par(mar=c(4,4,4,4))
barplot(table(df$das_class), main="Full dataset")
barplot(table(df[df$das_class != 0,]$das_class))
barplot(table(DATASET$das_class), main="Filtered (before 2016)")
barplot(table(DATASET[DATASET$das_class != 0,]$das_class))

# check for lognormal distribution (and compare vs Pareto): it looks more like the former.
qqnorm(DATASET$n_cit_3_log)
qex <- function(x) qexp((rank(x)-.375)/(length(x)+.25))
plot(qex(DATASET$n_cit_3),DATASET$n_cit_3_log)

##################
# BASELINES: OLS #
##################
# https://stats.idre.ucla.edu/r/dae/robust-regression/

require(MASS)
require(DMwR)

# OLS
summary(m_ols <- lm(n_cit_3_log ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = DATASET))
# controlling for journal too
#summary(m_ols <- lm(n_cit_3_log ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos + C(j_lower), data = DATASET))
# check residuals
opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(m_ols, las = 1)

# Robust OLS
summary(m_rols <- rlm(n_cit_3_log ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = DATASET))
# ANOVA
summary(m_aov <- aov(n_cit_3_log ~  n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = DATASET))

# Compare: OLS is fine
DMwR::regr.eval(DATASET$n_cit_3_log, m_ols$fitted.values)
DMwR::regr.eval(DATASET$n_cit_3_log, m_rols$fitted.values)
DMwR::regr.eval(DATASET$n_cit_3_log, m_aov$fitted.values)

# Output in LaTeX (Table 6)
require(stargazer)
stargazer(m_ols, m_rols, title="Results", align=TRUE, mean.sd = FALSE)

#########
# TOBIT #
#########
# https://stats.idre.ucla.edu/r/dae/tobit-models/
# Also see: http://www.stat.columbia.edu/~madigan/G6101/notes/logisticTobit.pdf 

require(VGAM)

summary(m <- vglm(n_cit_3_log ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, tobit(Lower = 0), data = DATASET))
ctable <- coef(summary(m))
pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(m), lower.tail = FALSE)
t <- cbind(ctable, pvals)
t

# significance of das_class via loglikelihood ratio test
m2 <- vglm(n_cit_3_log ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(journal_field) + das_required + das_encouraged + is_plos, tobit(Lower = 0), data = DATASET)
(p <- pchisq(2 * (logLik(m) - logLik(m2)), df = 2, lower.tail = FALSE))

# check residuals
DATASET$yhat <- fitted(m)[,1]
DATASET$rr <- resid(m, type = "response")
DATASET$rp <- resid(m, type = "pearson")[,1]

par(mfcol = c(2, 3))
par(mar=c(4,4,4,4))

with(DATASET, {
  plot(yhat, rr, main = "Fitted vs Residuals")
  qqnorm(rr)
  plot(yhat, rp, main = "Fitted vs Pearson Residuals")
  qqnorm(rp)
  plot(n_cit_3_log, rp, main = "Actual vs Pearson Residuals")
  plot(n_cit_3_log, yhat, main = "Actual vs Fitted")
})

# correlation predicted vs data
(r <- with(DATASET, cor(yhat, n_cit_3_log)))

############
# LOGISTIC #
############
# https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
# Really good results, yet the strong predictors here are year and PLoS due to policy timing effects.

require(nnet)

# Predicting if has_das or das_class
summary(m_logistic <- glm(has_das ~ n_cit_3_log + n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(journal_field) + das_required + das_encouraged + is_plos, data = DATASET, family = binomial))
summary(m_logistic <- multinom(C(das_class) ~ n_cit_3_log + n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(journal_field) + das_required + das_encouraged + is_plos, data = DATASET))
head(pp <- fitted(m_logistic)) # see probabilities

# Predicting if is_cited or not
summary(m_logistic <- glm(is_cited ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = DATASET, family=binomial(link='logit')))
anova(m_logistic, test="Chisq")

###################################
# GLM: NEGATIVE BINOMIAL and more #
###################################
# https://stats.idre.ucla.edu/r/dae/zinb/

require(MASS)
library(gamlss)

# standard negative binomial
summary(m_neg <- gamlss(n_cit_3 ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = na.omit(DATASET), family=NBF()))
# continuous lognormal
summary(m_log <- gamlss(n_cit_3 +1 ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = na.omit(DATASET), family=LOGNO()))
# Pareto type 2
summary(m_par <- gamlss(n_cit_3 +1 ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = na.omit(DATASET), family=PARETO2()))
# zero-inflated negative binomial
summary(m_zero_neg <- gamlss(n_cit_3 ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = na.omit(DATASET), family=ZINBF()))

# just with cited publications
summary(m_neg_neg <- glm.nb(n_cit_3 ~ n_authors + n_references_tot + p_year + p_month + h_index_mean + h_index_median + C(das_class) + C(journal_field) + das_required + das_encouraged + is_plos + C(das_class)*is_plos, data = df_filtered_negative))
ctable <- coef(summary(m_neg_neg))
pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(m_neg_neg), lower.tail = FALSE)
t2 <- cbind(ctable, pvals)
t2

# CITATIONS to packages
basecit <- system.file("CITATION", package="base")
source(basecit, echo=TRUE)
readCitationFile(basecit)

basecit <- system.file("CITATION", package="nnet")
source(basecit, echo=TRUE)
readCitationFile(basecit)

# Export BMC Journal DAS summary

library("dplyr")

df <- read.csv("dataset/export_full.csv", sep = ";")
j_df <- df %>%
  filter(is_bmc == 'True' & !is.na(j_lower)) %>%
  select(j_lower,das_class,has_das,das_encouraged,das_required) %>%
  mutate(
    j_lower = forcats::fct_explicit_na(j_lower),
    das_class = as.integer(das_class),
    has_das = as.integer(has_das)-1,
    das_encouraged = as.integer(das_encouraged)-1,
    das_required = as.integer(das_required)-1
  ) %>%
  group_by(j_lower,das_class) %>%
  summarise(N = n(),has_das = sum(has_das),das_encouraged = sum(das_encouraged) - sum(das_required),das_required = sum(das_required)) %>%
  as.data.frame()

write.csv(j_df, file = "dataset/journal_das_summary.csv",row.names=FALSE)
