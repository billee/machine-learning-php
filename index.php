<?php

require_once __DIR__ . '/vendor/autoload.php';

use Phpml\Metric\Regression;
use Phpml\Metric\Accuracy;
use Phpml\Dataset\CsvDataset;
use Phpml\Regression\SVR;
use Phpml\CrossValidation\StratifiedRandomSplit;

//loading the data

$data = new CsvDataset("./data/wine.csv", 13, true);

//preprocessing data
$dataset = new StratifiedRandomSplit($data, 0.2, 156);

// $dataset->getTrainSamples();
// $dataset->getTrainLabels();
// $dataset->getTestSamples();
// $dataset->getTestLabels();

//training
$regression = new SVR();
$regression->train($dataset->getTrainSamples(), $dataset->getTrainLabels());

$predicted = $regression->predict( $dataset->getTestSamples());

//evaluating machine learning models
$score = Regression::r2Score($dataset->getTestLabels(), $predicted);
echo "r2score is : ". $score .PHP_EOL;

foreach($predicted as &$target){
    $target = round($target,0);
}


$accuracy = Accuracy::score($dataset->getTestLabels(),$predicted);
echo 'accuracy is: '. $accuracy;

// //making predictions with trained models
// echo "\n";
// var_dump($regression->predict([80]));



