nn(fraud_net, [X], Y, [non_fraud, fraud]) :: fraud(X,Y).

0.001 :: rule(LittleAmount,Day,CategoryGroceryPos,fraud).
0.999 :: rule(LittleAmount,Day,CategoryGroceryPos,non_fraud).
0.001 :: rule(LittleAmount,Day,CategoryNonGroceryPos,fraud).
0.999 :: rule(LittleAmount,Day,CategoryNonGroceryPos,non_fraud).
0.001 :: rule(LittleAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.999 :: rule(LittleAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.001 :: rule(LittleAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.999 :: rule(LittleAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).

0.007 :: rule(LittleAmount,Night,CategoryGroceryPos,fraud).
0.993 :: rule(LittleAmount,Night,CategoryGroceryPos,non_fraud).
0.007 :: rule(LittleAmount,Night,CategoryNonGroceryPos,fraud).
0.993 :: rule(LittleAmount,Night,CategoryNonGroceryPos,non_fraud).


0.018 :: rule(MediumAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.982 :: rule(MediumAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).
0.018 :: rule(MediumAmount,Day,CategoryNonGroceryPos,fraud).
0.982 :: rule(MediumAmount,Day,CategoryNonGroceryPos,non_fraud).
0.018 :: rule(MediumAmount,Night,CategoryNonGroceryPos,fraud).
0.982 :: rule(MediumAmount,Night,CategoryNonGroceryPos,non_fraud).


0.108 :: rule(BigAmount,Day,CategoryGroceryPos,fraud).
0.892 :: rule(BigAmount,Day,CategoryGroceryPos,non_fraud).
0.108 :: rule(BigAmount,Day,CategoryNonGroceryPos,fraud).
0.892 :: rule(BigAmount,Day,CategoryNonGroceryPos,non_fraud).
0.108 :: rule(ExtremeAmount,Day,CategoryGroceryPos,fraud).
0.892 :: rule(ExtremeAmount,Day,CategoryGroceryPos,non_fraud).
0.108 :: rule(ExtremeAmount,Day,CategoryNonGroceryPos,fraud).
0.892 :: rule(ExtremeAmount,Day,CategoryNonGroceryPos,non_fraud).


0.849 :: rule(BigAmount,Night,CategoryGroceryPos,fraud).
0.151 :: rule(BigAmount,Night,CategoryGroceryPos,non_fraud).
0.849 :: rule(BigAmount,Night,CategoryNonGroceryPos,fraud).
0.151 :: rule(BigAmount,Night,CategoryNonGroceryPos,non_fraud).


0.963 :: rule(MediumAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.037 :: rule(MediumAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.963 :: rule(MediumAmount,Day,CategoryGroceryPos,fraud).
0.037 :: rule(MediumAmount,Day,CategoryGroceryPos,non_fraud).
0.963 :: rule(MediumAmount,Night,CategoryGroceryPos,fraud).
0.037 :: rule(MediumAmount,Night,CategoryGroceryPos,non_fraud).


0.448 :: rule(BigAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.552 :: rule(BigAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.448 :: rule(BigAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.552 :: rule(BigAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).
0.448 :: rule(ExtremeAmount,EarlyMorning,CategoryGroceryPos,fraud).
0.552 :: rule(ExtremeAmount,EarlyMorning,CategoryGroceryPos,non_fraud).
0.448 :: rule(ExtremeAmount,EarlyMorning,CategoryNonGroceryPos,fraud).
0.552 :: rule(ExtremeAmount,EarlyMorning,CategoryNonGroceryPos,non_fraud).


0.049 :: rule(ExtremeAmount,Night,CategoryGroceryPos,fraud).
0.951 :: rule(ExtremeAmount,Night,CategoryGroceryPos,non_fraud).
0.049 :: rule(ExtremeAmount,Night,CategoryNonGroceryPos,fraud).
0.951 :: rule(ExtremeAmount,Night,CategoryNonGroceryPos,non_fraud).

0.15 :: uniform_a(non_fraud) ; 0.85 :: uniform_a(fraud).
t(0.5) :: uniform_b(non_fraud) ; t(0.5) :: uniform_b(fraud).

outcome(fraud,fraud,fraud).
outcome(non_fraud,non_fraud,non_fraud).
outcome(fraud,non_fraud,Z) :- uniform_a(Z).
outcome(non_fraud,fraud,Z) :- uniform_b(Z).


predict_fraud(X,Amt,Hour,Category,Outcome) :-
    rule(Amt,Hour,Category,Z1),
    fraud(X,Z2),
    outcome(Z1,Z2,Outcome).

