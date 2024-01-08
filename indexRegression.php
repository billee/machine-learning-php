<?php

require_once __DIR__ . '/vendor/autoload.php';

use Phpml\Metric\Regression;
use Phpml\Dataset\CsvDataset;
use Phpml\Regression\LeastSquares;
use Phpml\CrossValidation\RandomSplit;

//loading the data

$data = new CsvDataset("./data/insurance.csv", 1, true);
//$data = new CsvDataset("./data/test.csv", 1, true);

//preprocessing data
$dataset = new RandomSplit($data, 0.2, 156);

// $dataset->getTrainSamples();
// $dataset->getTrainLabels();
// $dataset->getTestSamples();
// $dataset->getTestLabels();

//training
$regression = new LeastSquares();
$regression->train($dataset->getTrainSamples(), $dataset->getTrainLabels());

$predict = $regression->predict( $dataset->getTestSamples());

//evaluating machine learning models
$score = Regression::r2Score($dataset->getTestLabels(), $predict);
echo "r2score is : ". $score;  // % accuracy

//making predictions with trained models
echo "\n";
var_dump($regression->predict([80]));



