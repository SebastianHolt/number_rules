


# Load Libraries & Data ---------------------------------------------------------------
library('tidyverse')
library('ggplot2')
library('lme4')
library(lmerTest)
library(emmeans)
library(data.table)
library(ggplot2)

S <- read_csv(paste0("../data/processed_data/small.csv"))
L <- read_csv(paste0("../data/processed_data/large.csv"))
M <- read_csv(paste0("../data/processed_data/multi.csv"))

L$correct <- with(L, ifelse(is.na(response) | response == "", NaN, 
                              ifelse(expected == response, TRUE, FALSE)))

Z <- read_csv(paste0("../data/processed_data/monkey.csv"))
G <- read_csv(paste0("../data/processed_data/giraffe.csv"))


save_model_summary <- function(model, experiment, filename, message) {
  # Create file path for the model summary
  file_path <- paste0("../results/models/E", experiment, "/", filename, ".txt")
  
  # Open the file connection
  zz <- file(file_path, open = "wt")
  
  # Sink the message to the file
  sink(zz, type = "message", append = TRUE)
  message("#########################################\n")
  message(paste0(message, "\n\n#########################################\n"))
  sink(type = "message", append = TRUE)  # Close the message sink
  
  # Sink the summary output to the file
  sink(zz, type = "output", append = TRUE)
  print(summary(model))
  sink(append = TRUE)  # Close the output sink
  
  # Close the file connection
  close(zz)
}


###############################


# E1 A. Small Sets Task ---------------------------------------------------------------

# 1. Pre-registered model:
mS <- glmer(data=S, correct ~ cd1 + CP + expected + age + ( 1 | id ), family='binomial')
save_model_summary(mS,'1','1_small',"First, we'll look at the Small Sets task.")


# 2. Exploratory model, isolate CP and subset knowers:
S_SB <- S[S$CP == FALSE,]
S_CP <- S[S$CP == TRUE,]
mS_SB <- glmer(data=S_SB, correct ~ cd1 + expected + age + ( 1 | id ), family='binomial')
mS_CP <- glmer(data=S_CP, correct ~ cd1 + expected + age + ( 1 | id ), family='binomial')
save_model_summary(mS_SB,'1','2_small_SB',"Repeat the pre-registered analysis on only subset knowers")
save_model_summary(mS_CP,'1','2_small_CP',"Repeat the pre-registered analysis on only CP knowers")

# 3. Exploratory model, add CP:condition interaction:
mScp <- glmer(data=S, correct ~ cd1 * CP + expected + age + ( 1 | id ), family='binomial')
save_model_summary(mScp,'1','3_small_cpCond',"The pre-registered model, but see if condition interacts with CP knowledge")

# 4. Exploratory model, isolate SB, knower level as predictor:
S_SB$kl <- as.numeric(S_SB$kl)
mS_SBkl <- glmer(data=S_SB, correct ~ cd1 + expected + age + kl + ( 1 | id ), family='binomial')
save_model_summary(mS_SBkl,'1','4_small_kl',"For subset knowers, does knower level predict additive responses?")

# 5. Exploratory model, isolate SB, knower level sufficiency:
mS_SBsuf <- glmer(data=S_SB, correct ~ cd1 + expected + age + suf + ( 1 | id ), family='binomial')
save_model_summary(mS_SBsuf,'1','5_small_suf',"For subset knowers, does knowing the target predict additive responses?")


###############################


# E1 B. Large Sets Task ---------------------------------------------------------------

# 1. Pre-registered model, WITHOUT highest-count:
mL <- glmer(data=L, correct ~ cd1 + expected + age + ( 1 | id ), family='binomial')
save_model_summary(mL,'1','large',"Then, we'll look at the Large Sets task.")

###############################


# E1 C. Multiplier Task ---------------------------------------------------------------

# 1. Pre-registered model, WITHOUT highest-count:
mM <- glmer(data=M, correct ~ cd2 + expected + CP + age + ( 1 | id ) + ( 1 | cd1 ), family='binomial')
save_model_summary(mM,'1','multi',"Then, we'll look at the Multiplier task.")



# Save the above models out to file all in one place ---------------------------------------------------------------

path <- paste0("../results/models/E1/all.txt")
connect <- file(path, open = "wt")
sink(connect, type = "message")
message(paste0('All Small Sets:', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mS))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Subset only small sets', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mS_SB))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('CP only small sets', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mS_CP))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Check for CP:condition interaction', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mScp))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Look at subset-knower knower-level effect', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mS_SBkl))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Look knower-level sufficiency', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mS_SBsuf))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Large Sets task', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mL))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Multiplier task', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mM))
sink()  # Close the output sink
close(connect)

###############################

# Correlation of factors
cor(as.numeric(as.factor(S$CP)), S$age)

###### Try out the version with HC as a predictor
mS <- glmer(data=S, correct ~ cd1 + CP + expected + age + hc + ( 1 | id ), family='binomial')
mL <- glmer(data=L, correct ~ cd1 + expected + age + hc + ( 1 | id ), family='binomial')
mM <- glmer(data=M, correct ~ cd2 + expected + CP + age + hc + ( 1 | id ) + ( 1 | cd1 ), family='binomial')  # is singular

path <- paste0("../results/models/E1_HC/all.txt")
connect <- file(path, open = "wt")
sink(connect, type = "message")
message(paste0('All Small Sets:', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mS))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Large Sets task', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mL))
sink()  # Close the output sink

sink(connect, type = "message")
message("\n\n\n#########################################\n")
message(paste0('Multiplier task', "\n#########################################\n"))
sink(type = "message")  # Close the message sink
sink(connect, type = "output")
print(summary(mM))
sink()  # Close the output sink
close(connect)


##################



# E2 The Only Task ---------------------------------------------------------------

# 1. Pre-registered model:
m1 <- glmer(data = Z, correct ~ cond * giraffe_nums + target + training_acc + (1|ID), family='binomial')
save_model_summary(m1,'2','monkey',"First is the pre-registered model, including interaction.")

# 2. This exploratory analysis also predicts pseudo-English number compositions
m2 <- glmer(data = G, correct ~ cond * language + target + training_acc + (1|ID), family='binomial')
save_model_summary(m2,'2','giraffe',"Pivoting to predict Monkey AND Giraffe generalizations")

# 3. Did it matter how well kids remembered the unit words?
mGZ <- glmer(data = Z, correct ~ cond * giraffe_nums + target * gz_acc + (1|ID), family='binomial')
save_model_summary(mGZ,'2','memory',"Did accurate recall of unit words predict generalization accuracy?")

#############################


