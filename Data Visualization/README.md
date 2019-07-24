# Data Visualization: Titanic 1912 Tragedy #
 
## Summary ##
In 1912, the ship thought to be immortal sinked in its first voyage, hitting a glacier.
Approximately 2200 passengers were on board but data of almost 890 is captured.
The ship had 3 Passenger Classes (I, II, III) I being most elite one and hence forth.
Chances of Survival of Men and Women differed significantly with Women survivors peaking the count. Also survival of passengers across different passenger classes was higher for Class 1 and hence lowering down the order.

## Data and Design ##
The Visualization shows the chance of survival across different passenger classes and between men and women. The data points for the charts includes passenger class, gender, survival.
Data used for the respective visualization is clean, i cleared off unnecessary data and minor fixes for technical stability.
I made 3 version,V1:started with simple stacked bar comparison,then V2: addition of button to differentiate across gender and then V3: another button to bring clear pictures via Percentage button.

## Chart Selections ##
* Bar chart is easy to understand and good for comparison. Hence, a bar chart is used to show the differences across the passenger class.
* Stacked bar chart uses color to encode the category of data.Hence stacked bar charts are used to compare survivals.
* Interaction is used to one more dimension, the category of data aggregated by gender.
## Visual Encoding ##
* x position: passenger class.
* y position: number of passenger n.
* color hue: survived-red or perished-blue(as they perished in the blue sea).
* buttons: all passengers, males and females.
## Library Selection ##
* dimple.js is the primary visual library used in creating these charts. dimple.js is very easy to create charts and also has interactive charting features like hover.
* d3 is also used in this project. compared to dimple.js it is at lower level of abstraction. So it is not as efficient as dimple when creating charts, but it has more flexibility.
## Feedbacks ##
Feedbacks are collected on Version 1 of the Visualization and through in person interviews.The suggestion are considered for the Version 2 and later Version 3.
* Feedback 1 : Need a more clear and easy way to show Difference for Males vs Females. Adding a button would be nice and it will be fun to play with it too.
* Feedback 2 : It is not clear at the first glance which one is Survived and which is  Perished. Adding a legend would work wonders for this visualization as this visual will be understood in first look.
* Feedback 3 : There is just too much information in one visual, although i like the interaction i does need to maintain a mind palace to store all information.Keep it simple and straight. Nice use of color.
* Feedback 4(Udacity) : It would be better if you add percentage of passengers survived and perished on stacked bar across gender and Passenger class as to answer the questions like "What percent of women survived of Class 1?".
There is a Color flip for Men and Women for Survived and Perished.
Ordering of Passenger class on x-axis is confusing.

Separation of data will give viewers a breather.Also the headline “Titanic data analysis” gives viewers worry that they need to understand some analysis.
## Key Takeaways ##
   * I understood the importance of adding a button to separate out males with females to give users a martini glass approach as this is what they like. 
   * I added a legend for Survived v/s Perished giving the visual more credibility.
   * I changed the title to “Survived v/s Perished” as it is both simple, declares just one action and catchy.
   * For Version 3, as pointed out in Feedback 4,I made those changes:
   Introduced a percentage button for better understanding of data,
   Fixed Color Flip and Passenger class ordering.
## Resources ##
   * Titanic data set: https://www.kaggle.com/c/titanic-gettingStarted
   * Udacity Data Visualization with d3: http://udacity.com
