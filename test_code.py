import recommendations
#print recommendations.sim_distance(recommendations.critics, 'Lisa Rose', 'Gene Seymour')
#print recommendations.topMatches(recommendations.critics, 'Toby', n=3)
print recommendations.getRecommendations(recommendations.critics, 'Toby')
print recommendations.getRecommendations(recommendations.critics, 'Toby', similarity = recommendations.sim_distance)
