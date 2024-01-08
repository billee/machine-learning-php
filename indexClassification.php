<?php

require_once __DIR__ . '/vendor/autoload.php';

use Phpml\Metric\Regression;
use Phpml\Metric\Accuracy;
use Phpml\Dataset\CsvDataset;
use Phpml\Regression\LeastSquares;
use Phpml\CrossValidation\RandomSplit;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\Regression\SVR;
use Phpml\Classification\KNearestNeighbors;

//loading the data

//$data = new CsvDataset("./data/iris.csv", 4, true);
$data = new CsvDataset("./data/wine.csv", 13, true);


//preprocessing data
$dataset = new StratifiedRandomSplit($data, 0.2, 156);

// $dataset->getTrainSamples();
// $dataset->getTrainLabels();
// $dataset->getTestSamples();
// $dataset->getTestLabels();

//training
$classification = new KNearestNeighbors(3);
$classification->train($dataset->getTrainSamples(), $dataset->getTrainLabels());

$predict = $classification->predict( $dataset->getTestSamples());

//evaluating machine learning models
// $score = Regression::r2Score($dataset->getTestLabels(), $predict);
// echo "r2score is : ". $score. PHP_EOL; 

// foreach($predict as &$target){
//     $target =  round($target, 0);
// }

$accuracy = Accuracy::score($dataset->getTestLabels(), $predict);
echo "accuracy is : ". $accuracy;





