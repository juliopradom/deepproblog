:-use_module(library(lists)).

nn(fraud_net, [X], Y, [non_fraud, fraud]) :: fraud(X,Y).


0.001 :: rule(LittleAmount,Day,CategoryGroceryPos,fraud).
0.999 :: rule(LittleAmount,Day,CategoryGroceryPos,non_fraud).
0.001 :: rule(LittleAmount,Day,CategoryNonGroceryPos,fraud).
0.999 :: rule(LittleAmount,Day,CategoryNonGroceryPos,non_fraud).
0.001 :: rule(LittleAmount,Day,_,fraud).
0.999 :: rule(LittleAmount,Day,_,non_fraud).
0.001 :: rule(LittleAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.999 :: rule(LittleAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.001 :: rule(LittleAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.999 :: rule(LittleAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).
0.001 :: rule(LittleAmount,EarlyMorning,_,fraud).
0.999 :: rule(LittleAmount,EarlyMorning,_,non_fraud).

0.007 :: rule(LittleAmount,Night,CategoryGroceryPos,fraud).
0.993 :: rule(LittleAmount,Night,CategoryGroceryPos,non_fraud).
0.007 :: rule(LittleAmount,Night,CategoryNonGroceryPos,fraud).
0.993 :: rule(LittleAmount,Night,CategoryNonGroceryPos,non_fraud).
0.007 :: rule(LittleAmount,Night,_,fraud).
0.993 :: rule(LittleAmount,Night,_,non_fraud).
0.007 :: rule(LittleAmount,_,CategoryNonGroceryPos,fraud).
0.993 :: rule(LittleAmount,_,CategoryNonGroceryPos,non_fraud).
0.007 :: rule(LittleAmount,_,CategoryGroceryPos,fraud).
0.993 :: rule(LittleAmount,_,CategoryGroceryPos,non_fraud).
0.007 :: rule(LittleAmount,_,_,fraud).
0.993 :: rule(LittleAmount,_,_,non_fraud).


0.018 :: rule(MediumAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.982 :: rule(MediumAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).
0.018 :: rule(MediumAmount,EarlyMorning,_,fraud).
0.982 :: rule(MediumAmount,EarlyMorning,_,non_fraud).
0.018 :: rule(MediumAmount,Day,CategoryNonGroceryPos,fraud).
0.982 :: rule(MediumAmount,Day,CategoryNonGroceryPos,non_fraud).
0.018 :: rule(MediumAmount,Day,_,fraud).
0.982 :: rule(MediumAmount,Day,_,non_fraud).
0.018 :: rule(MediumAmount,Night,CategoryNonGroceryPos,fraud).
0.982 :: rule(MediumAmount,Night,CategoryNonGroceryPos,non_fraud).
0.018 :: rule(MediumAmount,Night,_,fraud).
0.982 :: rule(MediumAmount,Night,_,non_fraud).
0.018 :: rule(MediumAmount,_,CategoryNonGroceryPos,fraud).
0.982 :: rule(MediumAmount,_,CategoryNonGroceryPos,non_fraud).
t(0.5) :: rule(MediumAmount,_,_,fraud).
t(0.5) :: rule(MediumAmount,_,_,non_fraud).


0.108 :: rule(BigAmount,Day,CategoryGroceryPos,fraud).
0.892 :: rule(BigAmount,Day,CategoryGroceryPos,non_fraud).
0.108 :: rule(BigAmount,Day,CategoryNonGroceryPos,fraud).
0.892 :: rule(BigAmount,Day,CategoryNonGroceryPos,non_fraud).
0.108 :: rule(BigAmount,Day,_,fraud).
0.892 :: rule(BigAmount,Day,_,non_fraud).
0.108 :: rule(ExtremeAmount,Day,CategoryGroceryPos,fraud).
0.892 :: rule(ExtremeAmount,Day,CategoryGroceryPos,non_fraud).
0.108 :: rule(ExtremeAmount,Day,CategoryNonGroceryPos,fraud).
0.892 :: rule(ExtremeAmount,Day,CategoryNonGroceryPos,non_fraud).
0.108 :: rule(ExtremeAmount,Day,_,fraud).
0.892 :: rule(ExtremeAmount,Day,_,non_fraud).


0.849 :: rule(BigAmount,Night,CategoryGroceryPos,fraud).
0.151 :: rule(BigAmount,Night,CategoryGroceryPos,non_fraud).
0.849 :: rule(BigAmount,Night,CategoryNonGroceryPos,fraud).
0.151 :: rule(BigAmount,Night,CategoryNonGroceryPos,non_fraud).
0.849 :: rule(BigAmount,Night,_,fraud).
0.151 :: rule(BigAmount,Night,_,non_fraud).
t(0.5) :: rule(BigAmount,_,CategoryGroceryPos,fraud).
t(0.5) :: rule(BigAmount,_,CategoryGroceryPos,non_fraud).
t(0.5) :: rule(BigAmount,_,CategoryNonGroceryPos,fraud).
t(0.5) :: rule(BigAmount,_,CategoryNonGroceryPos,non_fraud).
t(0.5) :: rule(BigAmount,_,_,fraud).
t(0.5) :: rule(BigAmount,_,_,non_fraud).


0.963 :: rule(MediumAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.037 :: rule(MediumAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.963 :: rule(MediumAmount,Day,CategoryGroceryPos,fraud).
0.037 :: rule(MediumAmount,Day,CategoryGroceryPos,non_fraud).
0.963 :: rule(MediumAmount,Night,CategoryGroceryPos,fraud).
0.037 :: rule(MediumAmount,Night,CategoryGroceryPos,non_fraud).
0.963 :: rule(MediumAmount,_,CategoryGroceryPos,fraud).
0.037 :: rule(MediumAmount,_,CategoryGroceryPos,non_fraud).



0.448 :: rule(BigAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.552 :: rule(BigAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.448 :: rule(BigAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.552 :: rule(BigAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).
0.448 :: rule(BigAmount,EarlyMorning,_,fraud).
0.552 :: rule(BigAmount,EarlyMorning,_,non_fraud).
0.448 :: rule(ExtremeAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.552 :: rule(ExtremeAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.448 :: rule(ExtremeAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.552 :: rule(ExtremeAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).
0.448 :: rule(ExtremeAmount,EarlyMorning,_,fraud).
0.552 :: rule(ExtremeAmount,EarlyMorning,_,non_fraud).



0.049 :: rule(ExtremeAmount,Night,CategoryGroceryPos,fraud).
0.951 :: rule(ExtremeAmount,Night,CategoryGroceryPos,non_fraud).
0.049 :: rule(ExtremeAmount,Night,CategoryNonGroceryPos,fraud).
0.951 :: rule(ExtremeAmount,Night,CategoryNonGroceryPos,non_fraud).
0.049 :: rule(ExtremeAmount,Night,_,fraud).
0.951 :: rule(ExtremeAmount,Night,_,non_fraud).
t(0.5) :: rule(ExtremeAmount,_,CategoryGroceryPos,fraud).
t(0.5) :: rule(ExtremeAmount,_,CategoryGroceryPos,non_fraud).
t(0.5) :: rule(ExtremeAmount,_,CategoryNonGroceryPos,fraud).
t(0.5) :: rule(ExtremeAmount,_,CategoryNonGroceryPos,non_fraud).
t(0.5) :: rule(ExtremeAmount,_,_,fraud).
t(0.5) :: rule(ExtremeAmount,_,_,non_fraud).

t(0.5) :: rule(_,EarlyMorning,CategoryGroceryPos,fraud).
t(0.5) :: rule(_,EarlyMorning,CategoryGroceryPos,non_fraud).
t(0.5) :: rule(_,EarlyMorning,CategoryNonGroceryPos,fraud).
t(0.5) :: rule(_,EarlyMorning,CategoryNonGroceryPos,non_fraud).
t(0.5) :: rule(_,Day,CategoryGroceryPos,fraud).
t(0.5) :: rule(_,Day,CategoryGroceryPos,non_fraud).
t(0.5) :: rule(_,Day,CategoryNonGroceryPos,fraud).
t(0.5) :: rule(_,Day,CategoryNonGroceryPos,non_fraud).
t(0.5) :: rule(_,Night,CategoryGroceryPos,fraud).
t(0.5) :: rule(_,Night,CategoryGroceryPos,non_fraud).
t(0.5) :: rule(_,Night,CategoryNonGroceryPos,fraud).
t(0.5) :: rule(_,Night,CategoryNonGroceryPos,non_fraud).

t(0.5) :: rule(_,_,CategoryNonGroceryPos,fraud).
t(0.5) :: rule(_,_,CategoryNonGroceryPos,non_fraud).
t(0.5) :: rule(_,_,CategoryGroceryPos,fraud).
t(0.5) :: rule(_,_,CategoryGroceryPos,non_fraud).

t(0.5) :: rule(_,EarlyMorning,_,fraud).
t(0.5) :: rule(_,EarlyMorning,_,non_fraud).
t(0.5) :: rule(_,Day,_,fraud).
t(0.5) :: rule(_,Day,_,non_fraud).
t(0.5) :: rule(_,Night,_,fraud).
t(0.5) :: rule(_,Night,_,non_fraud).

0.1 :: rule(_,_,_,fraud).
0.9 :: rule(_,_,_,non_fraud).


0.15 :: uniform_a(non_fraud) ; 0.85 :: uniform_a(fraud).
t(0.5) :: uniform_b(non_fraud) ; t(0.5) :: uniform_b(fraud).


outcome(fraud,fraud,fraud).
outcome(non_fraud,non_fraud,non_fraud).
outcome(fraud,non_fraud,Z) :- uniform_a(Z).
outcome(non_fraud,fraud,Z) :- uniform_b(Z).

noise_handler(X,Amt,Hour,Category,is_noisy,Outcome) :-
    rule(Amt,Hour,Category,Outcome).

noise_handler(X,Amt,Hour,Category,is_not_noisy,Outcome) :-
    rule(Amt,Hour,Category,Z1),
    fraud(X,Z2),
    outcome(Z1,Z2,Outcome).

predict_fraud(X,Amt,Hour,Category,Noisy,Outcome) :-
    noise_handler(X,Amt,Hour,Category,Noisy,Outcome).


