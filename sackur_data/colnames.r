load("exp1.R")
head(data)

## suj: participant
## ses: ("session"): training = 1 / main = 2
## block: larger blocks (1 -- 10)
## miniblock: smaller blocks of 20 trials (1 -- 30)
## respside: (1/2) withinn participant counterbalancing of response side
## rule: -2, 0 (simple comparison), +2
## stim: stimulus (2, 4, 6, 8)
## acc: accuracy (0: error / 1: correct)
## rt: response times (ms --- values of 0 are probably anticipations)
## cg: congruency (1: congruent / -1 incongruent); is the result of the application of the rule one the same side of 5 as the stimulus itself? meaningful only for rule !=0
## target: result of the application of the rule to the stimulus
## code: ?
## begend: ?
## newCode: ? 

load("exp2.R")
head(data)

## cell: training (1: training on the cognitive operation at a 600 ms SOA / 2: training on the speeded response with 5 as a stimulus) or main experiment (0)
## ses: session (1 -- 5; one is the practice session);
## probe: probe SOA (the two values of 0 are bugs...)
## cresp: (correct response: name of the button "4" or "8" except for training 2)
## pont: ?
## resp: participant's response
## t2resp; t2rt: response and rt for training 2
## stimont: ?


load("exp3.R")
head(data)

## p: prime
## t: target

load("exp4_manual.R") 
head(data)

## x: cursor position (0 -- 20)
