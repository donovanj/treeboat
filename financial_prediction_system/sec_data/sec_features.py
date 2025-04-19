"""

sec_database> db.facts.aggregate([
...   {$match: {"facts.facts.us-gaap": {$exists: true}}},
...   {$project: {facts: {$objectToArray: "$facts.facts.us-gaap"}}},
...   {$unwind: "$facts"},
...   {$group: {_id: "$facts.k", count: {$sum: 1}}},
...   {$sort: {count: -1}},
...   {$limit: 40}
... ])
[
  { _id: 'OperatingLeaseRightOfUseAsset', count: 18 },
  { _id: 'IncomeTaxExpenseBenefit', count: 18 },
  { _id: 'NetCashProvidedByUsedInFinancingActivities', count: 18 },
  { _id: 'NetCashProvidedByUsedInInvestingActivities', count: 18 },
  { _id: 'NetCashProvidedByUsedInOperatingActivities', count: 18 },
  { _id: 'Assets', count: 18 },
  { _id: 'CashAndCashEquivalentsAtCarryingValue', count: 18 },
  { _id: 'OperatingLeaseLiability', count: 18 },
  { _id: 'LiabilitiesAndStockholdersEquity', count: 18 },
  { _id: 'EarningsPerShareDiluted', count: 17 },
  {
    _id: 'EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate',
    count: 17
  },
  { _id: 'PropertyPlantAndEquipmentNet', count: 17 },
  { _id: 'PaymentsToAcquireBusinessesNetOfCashAcquired', count: 17 },
  { _id: 'LesseeOperatingLeaseLiabilityPaymentsDue', count: 17 },
  { _id: 'ShareBasedCompensation', count: 17 },
  {
    _id: 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
    count: 17
  },
  { _id: 'EarningsPerShareBasic', count: 17 },
  { _id: 'AccumulatedOtherComprehensiveIncomeLossNetOfTax', count: 17 },
  { _id: 'NetIncomeLoss', count: 17 },
  { _id: 'Liabilities', count: 17 }
]
Type "it" for more
sec_database> it
[
  {
    _id: 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
    count: 17
  },
  { _id: 'PaymentsToAcquirePropertyPlantAndEquipment', count: 17 },
  { _id: 'DeferredIncomeTaxExpenseBenefit', count: 17 },
  { _id: 'ComprehensiveIncomeNetOfTax', count: 17 },
  {
    _id: 'AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount',
    count: 16
  },
  {
    _id: 'LesseeOperatingLeaseLiabilityPaymentsDueYearThree',
    count: 16
  },
  { _id: 'DeferredTaxAssetsValuationAllowance', count: 16 },
  { _id: 'WeightedAverageNumberOfDilutedSharesOutstanding', count: 16 },
  { _id: 'DeferredTaxAssetsGross', count: 16 },
  {
    _id: 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
    count: 16
  },
  { _id: 'RetainedEarningsAccumulatedDeficit', count: 16 },
  { _id: 'DeferredIncomeTaxLiabilities', count: 16 },
  { _id: 'FiniteLivedIntangibleAssetsNet', count: 16 },
  {
    _id: 'AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment',
    count: 16
  },
  { _id: 'WeightedAverageNumberOfSharesOutstandingBasic', count: 16 },
  {
    _id: 'LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths',
    count: 16
  },
  {
    _id: 'LesseeOperatingLeaseLiabilityUndiscountedExcessAmount',
    count: 16
  },
  { _id: 'GoodwillImpairmentLoss', count: 16 },
  { _id: 'UnrecognizedTaxBenefits', count: 16 },
  { _id: 'InterestPaidNet', count: 16 }
]

""" 
