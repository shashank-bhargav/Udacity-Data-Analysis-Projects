# Data Visualization: Titanic 1912 Tragedy #
 
## Summary ##
In 1912, the ship thought to be immortal sinked in its first voyage, hitting a glacier.
Approximately 2200 passengers were on board but data of almost 890 is captured.
The ship had 3 Passenger Classes (I, II, III) I being most elite one and hence forth.
Chances of Survival of Men and Women differed significantly at times.Hence, the following Analysis has been made.

## Data and Design ##
The Visualization shows the chance of survival across different passenger classes and between men and women. The data points for the charts includes passenger class, gender, survival.
Data used for the respective visualization is clean, i cleared off unnecessary data and minor fixes for technical stability.  
## Chart Selections ##
* Bar chart is easy to understand and good for comparison. Hence, a bar chart is used to show the differences across the passenger class.
* Stacked bar chart uses color to encode the category of data.Hence stacked bar charts are used to compare survivals.
* Interaction is used to one more dimension, the category of data aggregated by gender.
## Visual Encoding ##
* x position: passenger class.
* y position: number of passenger.
* color hue: survived or perished.
* buttons: all passengers, males and females.
## Library Selection ##
* dimple.js is the primary visual library used in creating these charts. dimple.js is very easy to create charts and also has interactive charting features like hover.
* d3 is also used in this project. compared to dimple.js it is at lower level of abstraction. So it is not as efficient as dimple when creating charts, but it has more flexibility.
## Feedbacks ##
Feedbacks are collected on Version 1 of the Visualization and through in person interviews.The suggestion are considered for the Version 2.
* Feedback 1 : Need a more clear and easy way to show Difference for Males vs Females. Adding a button would be nice and it will be fun to play with it too.
* Feedback 2 : It is not clear at the first glance which one is Survived and which is  Perished. Adding a legend would work wonders for this visualization as this visual will be understood in first look.
* Feedback 3 : There is just too much information in one visual, although i like the interaction i does need to maintain a mind palace to store all information.
Separation of data will give viewers a breather.Also the headline “Titanic data analysis” gives viewers worry that they need to understand some analysis.
Keep it simple and straight. Nice use of color.
## Key Takeaways ##
   * I understood the importance of adding a button to separate out males with females to give users a martini glass approach as this is what they like. 
   * I added a legend for Survived v/s Perished giving the visual more credibility.
   * I changed the title to “Survived v/s Perished” as it is both simple, declares just one action and catchy.
## Resources ##
   * Titanic data set: https://www.kaggle.com/c/titanic-gettingStarted
   * Udacity Data Visualization with d3: http://udacity.com
