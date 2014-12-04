
# p is the vector of predicted probabilities 
#and r is the vector of response values.

truepos  <- function(p, r)
        {
        threshold <- seq(0, 1, .01)
        if (is.logical(r))
        apply((apply(sapply(p, '>=', threshold), 1, '&',
        sapply(r, '==', T))), 2, sum)/sum(r == T)
        else 
        apply((apply(sapply(p, '>=', threshold), 1, '&',
        sapply(as.factor(r), '==', levels(as.factor(r))[2]))), 2, sum)/sum(as.factor(r) 		== levels(as.factor(r))[2])
        }
        
        
trueneg  <- function(p, r)
        {
        threshold <- seq(0, 1, .01)
        if (is.logical(r))
        apply((apply(sapply(p, '<=', threshold), 1, '&',
        sapply(r, '==', F))), 2, sum)/sum(r == F)
        else 
        apply((apply(sapply(p, '<=', threshold), 1, '&',
        sapply(as.factor(r), '==', levels(as.factor(r))[1]))), 2, sum)/sum(as.factor(r) 		== levels(as.factor(r))[1])
        }        

plot.roc  <- function(p, r, main = "ROC Curve", col = 4)
        {
        tp <- truepos(p, r)
        tn <- trueneg(p, r)
        plot(1-tn, tp,  type = "n", main = main,
        xlab = "False Positive", ylab = "True Positive")
		  lines(1-tn, tp, col = col, lwd =2)
        abline(0, 1)
        }

lines.roc  <- function(p, r, main = "ROC Curve", col)
        {
        tp <- truepos(p, r)
        tn <- trueneg(p, r)
		  lines(1-tn, tp, col = col, lwd =2)
        }



score.table = function(p, r, threshold = 0.5)
	{
		TH <- p > threshold
		Pred <- rep(0, length(TH))
		Pred[TH] <- 1
		Actual <- r
		cat("Predicted vs. Actual", fill = T)
		table(Pred, Actual)
	}
	
	
	
