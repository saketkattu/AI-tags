# Recommendation of Relavent Tags for Github Projects ( Based on ML-OPS)

## Objective and Current Situation
 We an application where users search for projects using tags.Our objective is to improve search and discoverability.


## Solution
1. <b>Visualze: </b> Ideal solution should ensure that all projects will have proper metadata(tags) so that users can discover them 
2. <b>Understand: </b>User have been search through only tags.Tags are more reliable to our models as they are given by the author of the project and tell us what the project is about . If the tags our explicity not mentioned then we can use other metadata to arrive at a tag 
3. <b>Design: </b>We would like all project to have appropriate tag,we have the necessary information (title, description, etc.) to meet that requirement 
    - Augment vs Automate : We will choose to augment the users rather to automate the process of adding tags 
    - UX contrains: Wil keep a track on the number of results(tags) suggested to the user as more tags will overwhelm and make the screen messy 
    - Tech Constrains: Need to maintain low latency (<10ms)

## Evaluation 
We need to suggest highly relevant tags (precision) so we don't fatigue the user with noise. But recall that the whole point of this task is to suggest tags that the author will miss (recall) so we can allow our users to find the best resource! So we'll need to tradeoff between precision and recall.

Our Evaluation metric will be <b>F1-score </b>


## Iteration
We will be using a rule-based approach to build a baseline and iterate through CNN,Tranformers and Atention based models 






