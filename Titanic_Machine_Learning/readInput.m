function [PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked] = readInput( M ) 

PassengerId = M(:,1);
Survived = M(:,2);
Pclass = M(:,3);
Name = M(:,4);
Sex = M(:,5);
Age = M(:,6);
SibSp = M(:,7);
Parch = M(:,8);
Ticket = M(:,9);
Fare = M(:,10);
Cabin = M(:,11);
Embarked = M(:,12);

end;