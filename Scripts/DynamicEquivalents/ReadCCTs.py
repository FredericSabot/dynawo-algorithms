from lxml import etree

CCTs = []
for i in range(50):
  root = etree.parse('Merged_TD_CCT_Random/It_%03d/aggregatedResults.xml' % i).getroot()
  CCTs.append(root[0].get('criticalTime'))
print(','.join(CCTs))

