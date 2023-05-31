library(tidyr)
library(dplyr)
library(tibble)
library(ez)
library(apaTables)
library(MBESS)
library(rstudioapi)

# Import filtered data: ----------------------------------------------------------#
current_path = rstudioapi::getActiveDocumentContext()$path 
print(dirname(dirname(current_path)))
setwd(dirname(dirname(current_path)))
wd <- getwd()
#data <- as.data.frame(read.csv("./results/probe_pair/ppexp_childes.csv",
data <- as.data.frame(read.csv("./results/probe_pair/probe_pair_emp.csv",
                               #data <- as.data.frame(read.csv("./results/probe_pair/ppexp_enwiki8.csv",
                               header=TRUE,
                               stringsAsFactors = TRUE)
)
data <- data %>%
  filter(given %in% c("AoA", "controlIncAoA")) %>%
  filter(month <= 24) %>%
  filter(pid < 100)


# Expt 1: Forced choice, AoA vs control-------------------------------------------#
data$pid <- paste(data$pid, data$given)
data$correct_choice <- as.logical(data$correct_choice)
monthwise_perf <- data %>%
  group_by(given, probe, n_given, pid) %>%
  summarise(correct_choice = mean(correct_choice))
monthwise_perf$n_given <- as.factor(monthwise_perf$n_given)

# One-sample t-tests for performance relative to chance (Table S1) ---------------#
n_givens <- list()
p_ctrl <- list()
t_ctrl <- list()
ci_ctrl_lo <- list()
ci_ctrl_hi <- list()


for (n in unique(monthwise_perf$n_given)){
  mth_ctrl <- monthwise_perf %>%
    filter(n_given == n) %>%
    filter(given == "AoA")
  
  t_t_ctrl <- t.test(mth_ctrl$correct_choice, mu=0.5)
  n_givens <- append(n_givens, n)
  p_ctrl <- append(p_ctrl, t_t_ctrl$p.value)
  t_ctrl <- append(t_ctrl, t_t_ctrl$statistic)
  ci_ctrl_lo <- append(ci_ctrl_lo, t_t_ctrl$conf.int[1])
  ci_ctrl_hi <- append(ci_ctrl_hi, t_t_ctrl$conf.int[2])
}

df_aoa <- as.data.frame(cbind(n_givens, p_ctrl, t_ctrl, ci_ctrl_lo, ci_ctrl_hi))

n_givens <- list()
p_ctrl <- list()
t_ctrl <- list()
ci_ctrl_lo <- list()
ci_ctrl_hi <- list()
for (n in unique(monthwise_perf$n_given)){
  mth_ctrl <- monthwise_perf %>%
    filter(n_given == n) %>%
    filter(given == "controlIncAoA")
  
  t_t_ctrl <- t.test(mth_ctrl$correct_choice, mu=0.5)
  n_givens <- append(n_givens, n)
  p_ctrl <- append(p_ctrl, t_t_ctrl$p.value)
  t_ctrl <- append(t_ctrl, t_t_ctrl$statistic)
  ci_ctrl_lo <- append(ci_ctrl_lo, t_t_ctrl$conf.int[1])
  ci_ctrl_hi <- append(ci_ctrl_hi, t_t_ctrl$conf.int[2])
}

df_control <- as.data.frame(cbind(n_givens, p_ctrl, t_ctrl, ci_ctrl_lo, ci_ctrl_hi))



# Repeated measures ANOVA - accuracy (Table S2) ---------------------------------#
rm.anova.acc.ez <- ezANOVA(
  data=monthwise_perf,
  dv=.(correct_choice),
  wid=.(pid),
  within=.(n_given, probe),
  between=.(given),
  type=2
)

#apa.ezANOVA.table(rm.anova.acc.ez, correction="none") # Results table - accuracy

loweretasquared <- c()
upperetasquared <- c()
partialetasquared <- c()
for (cR in 1:nrow(rm.anova.acc.ez$ANOVA)) {
  Lims <- conf.limits.ncf(
    F.value = rm.anova.acc.ez$ANOVA$F[cR],
    conf.level = 0.95,
    df.1 <- rm.anova.acc.ez$ANOVA$DFn[cR],
    df.2 <- rm.anova.acc.ez$ANOVA$DFd[cR])
  Lower.lim <- Lims$Lower.Limit/(Lims$Lower.Limit + df.1 + df.2 + 1)
  Upper.lim <- Lims$Upper.Limit/(Lims$Upper.Limit + df.1 + df.2 + 1)
  if (is.na(Lower.lim)) {
    Lower.lim <- 0
  }
  if (is.na(Upper.lim)) {
    Upper.lim <- 1
  }
  loweretasquared <- c(loweretasquared,Lower.lim)
  upperetasquared <- c(upperetasquared,Upper.lim)
  partialetasquared <- c(partialetasquared,(Upper.lim + Lower.lim)/2)
}
rm.anova.acc.ez$ANOVA$lower_pes <- loweretasquared
rm.anova.acc.ez$ANOVA$upper_pes <- upperetasquared
rm.anova.acc.ez$ANOVA$pes <- partialetasquared
print(rm.anova.acc.ez$ANOVA)


# Expt 2: Forced choice results including generative models----------------------#

data2 <- as.data.frame(read.csv(
  "./results/probe_pair/probe_pair_optimal.csv",
  header=TRUE,
  stringsAsFactors = TRUE))

data3 <- as.data.frame(read.csv(
  "./results/probe_pair/probe_pair_probmatch.csv",
  header=TRUE,
  stringsAsFactors = TRUE))

data <- rbind(data, data2)
data <- rbind(data, data3)

data <- data %>%
  filter(n_given > 0)

data$pid <- paste(data$pid, data$given)
data$correct_choice <- as.logical(data$correct_choice)
monthwise_perf <- data %>%
  group_by(given, probe, n_given, pid) %>%
  summarise(correct_choice = mean(correct_choice))
monthwise_perf$n_given <- as.factor(monthwise_perf$n_given)

# Comparing structural models to AoA (Table S4) --------------------------------#
n_givens <- list()
p_ctrl <- list()
t_ctrl <- list()
ci_ctrl_lo <- list()
ci_ctrl_hi <- list()
probe_c <- list()
comp <- list()

for (p in c("control", "AoA")){
  for (n in unique(monthwise_perf$n_given)){
    mth_prob <- monthwise_perf %>%
      filter(n_given == n) %>%
      filter(probe == p) %>%
      filter(given == "synthProbIncAoA")
    
    mth_opt <- monthwise_perf %>%
      filter(n_given == n) %>%
      filter(probe == p) %>%
      filter(given == "synthOptIncAoA")
    
    t_t_struccomp <- t.test(mth_prob$correct_choice, mth_opt$correct_choice)
    n_givens <- append(n_givens, n)
    p_ctrl <- append(p_ctrl, t_t_struccomp$p.value)
    t_ctrl <- append(t_ctrl, t_t_struccomp$statistic)
    ci_ctrl_lo <- append(ci_ctrl_lo, t_t_struccomp$conf.int[1])
    ci_ctrl_hi <- append(ci_ctrl_hi, t_t_struccomp$conf.int[2])
    probe_c <- append(probe_c, p)
    comp <- append(comp, "matchprob-opt")
  }
}

for (p in c("control", "AoA")){
  for (n in unique(monthwise_perf$n_given)){
    mth_prob <- monthwise_perf %>%
      filter(n_given == n) %>%
      filter(probe == p) %>%
      filter(given == "synthProbIncAoA")
    
    mth_opt <- monthwise_perf %>%
      filter(n_given == n) %>%
      filter(probe == p) %>%
      filter(given == "AoA")
    
    t_t_struccomp <- t.test(mth_prob$correct_choice, mth_opt$correct_choice)
    n_givens <- append(n_givens, n)
    p_ctrl <- append(p_ctrl, t_t_struccomp$p.value)
    t_ctrl <- append(t_ctrl, t_t_struccomp$statistic)
    ci_ctrl_lo <- append(ci_ctrl_lo, t_t_struccomp$conf.int[1])
    ci_ctrl_hi <- append(ci_ctrl_hi, t_t_struccomp$conf.int[2])
    probe_c <- append(probe_c, p)
    comp <- append(comp, "matchprob-aoa")
    
  }
}

for (p in c("control", "AoA")){
  for (n in unique(monthwise_perf$n_given)){
    mth_prob <- monthwise_perf %>%
      filter(n_given == n) %>%
      filter(probe == p) %>%
      filter(given == "AoA")
    
    mth_opt <- monthwise_perf %>%
      filter(n_given == n) %>%
      filter(probe == p) %>%
      filter(given == "synthOptIncAoA")
    
    t_t_struccomp <- t.test(mth_prob$correct_choice, mth_opt$correct_choice)
    n_givens <- append(n_givens, n)
    p_ctrl <- append(p_ctrl, t_t_struccomp$p.value)
    t_ctrl <- append(t_ctrl, t_t_struccomp$statistic)
    ci_ctrl_lo <- append(ci_ctrl_lo, t_t_struccomp$conf.int[1])
    ci_ctrl_hi <- append(ci_ctrl_hi, t_t_struccomp$conf.int[2])
    probe_c <- append(probe_c, p)
    comp <- append(comp, "AoA-opt")
    
  }
}

df_struccomp <- as.data.frame(
  cbind(n_givens, p_ctrl, t_ctrl, ci_ctrl_lo, ci_ctrl_hi, probe_c, comp)
)
df_struccomp$sig_bonferroni <- df_struccomp$p_ctrl < 0.05/length(p_ctrl)
print(length(df_struccomp))