package liborInArrearsLiborMarketModel;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import liborInArrearsLiborMarketModel.LIBORMarketModelConstructionWithDynamicsAndMeasureSpecification.Dynamics;
import liborInArrearsLiborMarketModel.LIBORMarketModelConstructionWithDynamicsAndMeasureSpecification.Measure;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulationModel;
import net.finmath.montecarlo.interestrate.TermStructureModel;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.plots.Plot;
import net.finmath.plots.Plots;
import net.finmath.time.TimeDiscretization;



public class LiborInArrearsTest {

	private final static DecimalFormat formatterDouble = new DecimalFormat("0.0000");
	private final static DecimalFormat formatterDeviation = new DecimalFormat("0.000%;");

	final int	numberOfPaths	= 12000;
	//parameters for the two time discretizations
	final double simulationTimeStep = 0.1;
	final double LIBORTimeStep = 0.5;
	final int LIBORRateTimeHorizon = 16;
	final double notional = 1000;

	//fixing times for the forwards: the forwards corresponding to other fixing times will be interpolated
	final double[] fixingForGivenForwards = { 0.5, 1.0, 2.0, 3.0};
	final double[] forwardsForCurve = { 0.05, 0.05, 0.05, 0.05};

	final double correlationDecayParameter = 0.5;

	final double a = 0.1, b = 0.1, c = 0.15, d = 0.15; //volatility structure

	public void testMeasureSpotAndTerminalForInArrears() throws Exception {


		// We test it for a lognormal model, i.e we will use the analytical value within the Black Model to have a Benchmark value for the error
		Dynamics dynamics = Dynamics.LOGNORMAL;

		final   LIBORModelMonteCarloSimulationModel myLiborModelMonteCarloTerminal =
				LIBORMarketModelConstructionWithDynamicsAndMeasureSpecification.createLIBORMarketModel(
						numberOfPaths,
						simulationTimeStep,
						LIBORTimeStep, //T_i-T_{i-1}, we suppose it to be fixed
						LIBORRateTimeHorizon, //T_n
						fixingForGivenForwards,
						forwardsForCurve,
						correlationDecayParameter, // decay of the correlation between LIBOR rates
						dynamics,
						Measure.TERMINAL, // first terminal
						a, b, c, d,
						1897 // seed
						);

		final   LIBORModelMonteCarloSimulationModel myLiborModelMonteCarloSpot =
				LIBORMarketModelConstructionWithDynamicsAndMeasureSpecification.createLIBORMarketModel(
						numberOfPaths,
						simulationTimeStep,
						LIBORTimeStep, //T_i-T_{i-1}, we suppose it to be fixed
						LIBORRateTimeHorizon, //T_n
						fixingForGivenForwards,
						forwardsForCurve,
						correlationDecayParameter, // decay of the correlation between LIBOR rates
						dynamics,
						Measure.SPOT, // then spot
						a, b, c, d,
						1897 //seed
						);

		// This one is only for extracting the variance of the model for the analytic formula to be used for get the variance
		final TimeDiscretization simulationTimeDiscretization =  myLiborModelMonteCarloTerminal.getBrownianMotion().getTimeDiscretization();

		final TermStructureModel liborModel = myLiborModelMonteCarloTerminal.getModel();

		//	getIntegratedLIBORCovariance() is defined in LIBORMarketModel: we need to downcast
		final double[][][] integratedVarianceMatrix = ((LIBORMarketModel) liborModel).
				getIntegratedLIBORCovariance(simulationTimeDiscretization);
		//
		//		extract the discount curve (i.e., the zero coupon bonds curve) in order to get the analytical price
		final DiscountCurve discountFactors = liborModel.getDiscountCurve();
		//
		//		//extract the forward curve (i.e., the Libor curve) in order to get the analytical price
		final ForwardCurve forwards = liborModel.getForwardRateCurve();

		List<Double> fixings = new ArrayList<Double>();
		List<Double> errorsTerm = new ArrayList<Double>();
		List<Double> errorsSpot = new ArrayList<Double>();


		System.out.println("Price of Libor Rate in Arrears:\n");

		System.out.println("PaymentDate:     SimuTerminal:     SimuSpot:    Analytic:     RelDifTerminal:       RelDifSpot:        StdErrorTerminal:       StdErrorSpot: \n");


		// We will investigate in the libors with different payment/fixing
		for(int periodStart = 0 ; periodStart <= myLiborModelMonteCarloTerminal.getNumberOfLibors() - 1 ; periodStart += 1)

		{
			// Construct the product
			final double liborPeriodTi = myLiborModelMonteCarloTerminal.getLiborPeriod(periodStart);//T_i
			final double liborPeriodTiPlusOne = myLiborModelMonteCarloTerminal.getLiborPeriod(periodStart + 1); // T_i+1
			System.out.print(formatterDouble.format(liborPeriodTi ) + "            ");

			// Create Product
			final AbstractLIBORMonteCarloProduct liborInArrears = new LiborInArrears(liborPeriodTi, liborPeriodTiPlusOne);

			// Value under Monte-Carlo Simulation with Terminal and Spot Measure
			final double valueSimulationTerminal = notional * liborInArrears.getValue(myLiborModelMonteCarloTerminal);
			final double valueSimulationSpot = notional * liborInArrears.getValue(myLiborModelMonteCarloSpot);

			System.out.print(formatterDouble.format(valueSimulationTerminal) + "          ");
			System.out.print(formatterDouble.format(valueSimulationSpot) + "       ");

			//	Input for our formula : Variance, Bond in Fixing Time T_i, Bond in PeriodEnd T_i+1 , Initial Libor
			final int maturityIndexInTheSimulationDiscretization = myLiborModelMonteCarloTerminal.getTimeIndex(liborPeriodTi);
			final double integratedVariance = integratedVarianceMatrix
					[maturityIndexInTheSimulationDiscretization][periodStart][periodStart];
			final double variance = integratedVariance/liborPeriodTi ; // extract case T_0 = 0 -> divide by zero
			//		final double standardDeviation = Math.sqrt(variance); // No need since we directly put in the variance, i.e sigma^2

			final double zeroBondInFixing = discountFactors.getDiscountFactor(liborPeriodTi); // P(T_i,0)
			final double zeroBondInEndPeriod = discountFactors.getDiscountFactor(liborPeriodTiPlusOne); // P(T_i+1,0)

			final double initialForwardLibor = forwards.getForward(null, liborPeriodTi); // L(T_i,T_i+1,0)

			//Initialize
			double valueAnalytic = 0.0;

			// Extract case T_0 = 0 for the analytic value
			if( liborPeriodTi == 0) {
				valueAnalytic = LIBORTimeStep * initialForwardLibor  ;
			}
			else {
				valueAnalytic = valueAnalytic + LiborInArrearsAnalyticFormula.calculateLiborInArrearsFloaterAnalytic(initialForwardLibor,
						variance, liborPeriodTi, liborPeriodTiPlusOne, zeroBondInEndPeriod, zeroBondInFixing);
			}
			final double valueAnalytic2 = notional * valueAnalytic ;
			System.out.print(formatterDouble.format(valueAnalytic2) + "       ");

			//Relative Difference to the analytical Value
			final double relativeDifferenceTerminal = Math.abs(valueSimulationTerminal - valueAnalytic2)/valueAnalytic2;
			final double relativeDifferenceSpot = Math.abs(valueSimulationSpot - valueAnalytic2)/valueAnalytic2;
			System.out.print(formatterDeviation.format(relativeDifferenceTerminal) + "                ");
			System.out.print(formatterDeviation.format(relativeDifferenceSpot) + "               ");

			// Standard Error to see MC-Error
			final double valueSimulationTerminalStandardError =  liborInArrears.getValue(0.0, myLiborModelMonteCarloTerminal).getStandardError();
			System.out.print(formatterDouble.format(valueSimulationTerminalStandardError) + "              "   );

			final double valueSimulationSpotStandardError =  liborInArrears.getValue(0.0, myLiborModelMonteCarloSpot).getStandardError();
			System.out.println(formatterDouble.format(valueSimulationSpotStandardError) );

			fixings.add(liborPeriodTi);

			errorsTerm.add(relativeDifferenceTerminal);

			errorsSpot.add(relativeDifferenceSpot);
			//Assert.assertTrue(relativeDifference < tolerance);
		}

		Plot plotTerm = Plots.createScatter(fixings, errorsTerm, 0.0, 0.01, 4)
				.setTitle("Libor in Arrears error when using terminal measure" )
				.setXAxisLabel("fixing")
				.setYAxisLabel("MC-Error")
				.setYAxisNumberFormat(new DecimalFormat("0.0E00"));

		plotTerm.show();

		Plot plotSpot = Plots.createScatter(fixings, errorsSpot, 0.0, 0.01, 4)
				.setTitle("Libor in Arrears error when using spot measure" )
				.setXAxisLabel("fixing")
				.setYAxisLabel("MC-Error")
				.setYAxisNumberFormat(new DecimalFormat("0.0E00"));

		plotSpot.show();
	}


	public static void main(String[] args)  throws Exception {
		(new LiborInArrearsTest()).testMeasureSpotAndTerminalForInArrears();
	}

}
