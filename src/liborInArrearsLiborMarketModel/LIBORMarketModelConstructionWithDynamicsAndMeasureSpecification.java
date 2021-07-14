package liborInArrearsLiborMarketModel;

import java.util.HashMap;
import java.util.Map;

import net.finmath.exception.CalculationException;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveFromForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterpolation;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.RandomVariableFromArrayFactory;
import net.finmath.montecarlo.interestrate.CalibrationProduct;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulationModel;
import net.finmath.montecarlo.interestrate.LIBORMonteCarloSimulationFromLIBORModel;
import net.finmath.montecarlo.interestrate.models.LIBORMarketModelFromCovarianceModel;
import net.finmath.montecarlo.interestrate.models.covariance.AbstractLIBORCovarianceModel;
import net.finmath.montecarlo.interestrate.models.covariance.BlendedLocalVolatilityModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCorrelationModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCovarianceModelFromVolatilityAndCorrelation;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORVolatilityModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORVolatilityModelFromGivenMatrix;
import net.finmath.montecarlo.model.ProcessModel;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * This class creates a LIBOR market model, basing on the classes of the Finmath library. Here the user can specify if
 * the simulation has to be created under log-normal or normal dynamics, and if the measure considered has to be
 * the terminal or the spot one.
 *
 *
 */
public class LIBORMarketModelConstructionWithDynamicsAndMeasureSpecification {

	public enum Measure				{ SPOT, TERMINAL };
	public enum Dynamics			{ NORMAL, LOGNORMAL };

	/**
	 * It specifies and creates a Rebonato volatility structure, represented by a matrix, for the LIBOR
	 * Market Model. In particular, we have
	 * dL_i(t_j)=\sigma_i(t_j)L_i(t_j)dW_i(t_j)
	 * with
	 * \sigma_i(t_j)=(a+b(T_i-t_j))\exp(-c(T_i-t_j))+d,
	 * for t_j < T_i,
	 * for four parameters a,b,c,d with b,c>0
	 * @param a
	 * @param b
	 * @param c
	 * @param d
	 * @param simulationTimeDiscretization, the time discretization for the evolution of the processes
	 * @param tenureStructureDiscretization, the tenure structure T_0 < T_1< ...<T_n
	 * @return the matrix that represents the volatility structure: volatility[i,j]=sigma_j(t_i)
	 */
	private static double[][] createVolatilityStructure(double a, double b,double c, double d,
			TimeDiscretization simulationTimeDiscretization,
			TimeDiscretization tenureStructureDiscretization,
			Dynamics dynamics
			) {
		//volatility[i,j]=sigma_j(t_i)
		final int numberOfSimulationTimes = simulationTimeDiscretization.getNumberOfTimeSteps();
		final int numberOfTenureStructureTimes = tenureStructureDiscretization.getNumberOfTimeSteps();
		final double[][] volatility = new double[numberOfSimulationTimes][numberOfTenureStructureTimes];

		for (int timeIndex = 0; timeIndex < numberOfSimulationTimes; timeIndex++) {
			for (int LIBORIndex = 0; LIBORIndex < numberOfTenureStructureTimes; LIBORIndex++) {
				final double currentTime = simulationTimeDiscretization.getTime(timeIndex);//t_j
				final double currentMaturity = tenureStructureDiscretization.getTime(LIBORIndex);//T_i
				final double timeToMaturity = currentMaturity - currentTime;
				double instVolatility;
				if (timeToMaturity <= 0) {
					instVolatility = 0; // This forward rate is already fixed, no volatility
				}
				else {
					instVolatility = d + (a + b * timeToMaturity)
							* Math.exp(-c * timeToMaturity);//\sigma_i(t)=(a+b(T_i-t))\exp(-c(T_i-t))+d
				}
				if (dynamics == Dynamics.NORMAL) {
					instVolatility *= 0.05;
				}
				// Store
				volatility[timeIndex][LIBORIndex] = instVolatility;
			}
		}
		return volatility;
	}
	/**
	 * It simulates a LIBOR Market Model, by using the implementation of the Finmath library.
	 * @param numberOfPaths: number of simulations
	 * @param simulationTimeStep: the time step for the simulation of the LIBOR processes
	 * @param LIBORPeriodLength: the length of the interval between times of the tenure structure
	 * @param LIBORRateTimeHorizon: final LIBOR maturity
	 * @param fixingForGivenForwards: the times of the tenure structure where the initial
	 * forwards (also called LIBORs if you want, here we stick to the name of the Finmath library) are given
	 * @param givenForwards: the given initial forwards (from which the others are interpolated)
	 * @param correlationDecayParam, parameter \alpha>0, for the correlation of the LIBORs: in particular, we have
	 * dL_i(t_j)=\sigma_i(t_j)L_i(t_j)dW_i(t_j)
	 * with
	 * d<W_i,W_k>(t)= \rho_{i,k}(t)dt
	 *  where
	 * \rho_{i,j}(t)=\exp(-\alpha|T_i-T_k|)
	 * @param a, the first term for the volatility structure: the volatility in the SDEs above is given by
	 * \sigma_i(t_j)=(a+b(T_i-t_j))\exp(-c(T_i-t_j))+d,
	 * for t_j < T_i.
	 * @param b, the second term for the volatility structure
	 * @param c, the third term for the volatility structure
	 * @param d, the fourth term for the volatility structure
	 * @return an object implementing LIBORModelMonteCarloSimulationModel, i.e., representing the simulation of a LMM
	 * @throws CalculationException
	 */
	public static final LIBORModelMonteCarloSimulationModel createLIBORMarketModel(int numberOfPaths,
			double simulationTimeStep,
			double LIBORPeriodLength, //T_i-T_{i-1}, we suppose it to be fixed
			double LIBORRateTimeHorizon, //T_n
			double[] fixingForGivenForwards,
			double[] givenForwards,
			double correlationDecayParam, // decay of the correlation between LIBOR rates
			Dynamics dynamics,
			Measure measureType,
			double a, double b, double c, double d,
			int seed
			)
					throws CalculationException {
		/*
		 In order to simulate a LIBOR market model, we need to proceed along the following steps:
		 1) provide the time discretization for the evolution of the processes
		 2) provide the time discretization of the tenure structure
		 3) provide the observed term structure of the initial LIBOR rates (also called forwards, using the terminology
		 of the Finmath library) and if needed interpolate the ones missing: in this way we obtain the initial values
		 for the LIBOR processes
		 4) create the volatility structure, i.e., the terms sigma_i(t_j) in
		 	dL_i(t_j)=\sigma_i(t_j)L_i(t_j)dW_i(t_j)
		 	or
		 	dL_i(t_j)=\sigma_i(t_j)dW_i(t_j)

	 	 5) create the correlation structure, i.e., define the terms \rho_{i,j}(t) such that
	 	 d<W_i,W_k>(t)= \rho_{i,k}(t)dt
		 6) combine all steps 1, 2, 4, 5 to create a covariance model
		 7) give the covariance model to the constructor of BlendedLocalVolatilityModel, to get another covariance model that now
		 takes into account if the dynamics are normal (in this case the volatility gets rescaled) or log-normal
		 8) combine steps 2, 3, 7 to create the LIBOR model, also adding the appropriate properties about dynamics and measure
		 9) create a Euler discretization of the model we defined in step 8, specifying the model itself and
		 a Brownian motion that uses the time discretization defined in step 1
		 9) give the Euler scheme to the constructor of LIBORMonteCarloSimulationFromLIBORModel, to create an object of
		 type LIBORModelMonteCarloSimulationModel
		 */

		// Step 1: create the time discretization for the simulation of the processes
		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0, (int) (LIBORRateTimeHorizon / simulationTimeStep), simulationTimeStep);

		// Step 2: create the time discretization for the tenure structure (i.e., the dates T_1,..,T_n)
		final TimeDiscretization LIBORPeriodDiscretization = new
				TimeDiscretizationFromArray(0.0, (int) (LIBORRateTimeHorizon /LIBORPeriodLength), LIBORPeriodLength);

		/*
		  Step 3 Create the forward curve (initial values for the LIBOR market model). We suppose
		  not to have all the forwards: the others are interpolated using the specific method
		  of the Finmath library
		 */
		final ForwardCurve forwardCurve = ForwardCurveInterpolation.createForwardCurveFromForwards(
				"forwardCurve", // name of the curve
				fixingForGivenForwards, // fixings of the forward
				givenForwards, // the forwards we have
				LIBORPeriodLength
				);

		final DiscountCurve discountCurve = new DiscountCurveFromForwardCurve(forwardCurve);

		// Step 4, the volatility model: we only have to provide the matrix
		final double[][] volatility = createVolatilityStructure(
				a, b, c, d,
				timeDiscretization,
				LIBORPeriodDiscretization, dynamics);

		final LIBORVolatilityModel volatilityModel =
				new LIBORVolatilityModelFromGivenMatrix(timeDiscretization,
						LIBORPeriodDiscretization, volatility);
		/*
		  Step 5
		  Create a correlation model rho_{i,j} = exp(−a ∗ |T_i −T_j|)
		 */
		final LIBORCorrelationModel correlationModel =
				new LIBORCorrelationModelExponentialDecay(
						timeDiscretization,
						LIBORPeriodDiscretization,
						LIBORPeriodDiscretization.getNumberOfTimes()-1,//no factor reduction for now
						correlationDecayParam);

		/*
		 Step 6
		 Combine volatility model and correlation model, together with the two time discretizations,
		 to get a covariance model
		 */
		final LIBORCovarianceModelFromVolatilityAndCorrelation covarianceModel =
				new LIBORCovarianceModelFromVolatilityAndCorrelation(
						timeDiscretization,
						LIBORPeriodDiscretization,
						volatilityModel,
						correlationModel);

		//first we check if the dynamics are log-normal
		final boolean isLogNormal = (dynamics == Dynamics.LOGNORMAL);

		/*
		 * Step 7
		 * Here you substitute your covariance model with a new one built on top of it. The new model is given by
		 * (a L0 + (1-a)L) F
		 * where a=parameterForBlended is the displacement parameter, L is the component of the stochastic process, L_0 is its value
		 * at time zero and F is the factor loading from the given covariance model.
		 * Here the point is parameterForBlended: we construct it is such a way that is 0 if the dynamics are log-normal (so in this
		 * case nothing happens) and 1 if they are normal (in this case, the volatility gets rescaled multiplying it by the initial
		 * value of our processes).
		 *
		 */

		final double parameterForBlended = isLogNormal ? 0.0 : 1.0;

		final AbstractLIBORCovarianceModel covarianceModelBlended = new BlendedLocalVolatilityModel(
				covarianceModel, forwardCurve, parameterForBlended, false);
		//d\bar L = \bar L sigma dW

		//final AbstractLIBORCovarianceModel covarianceModelBlended = covarianceModel;

		//Step 8: we now create the model (i.e., the object of type LiborMarketModel)
		// Set model properties
		final Map<String, String> properties = new HashMap<>();

		final boolean isTerminal = (measureType == Measure.TERMINAL);

		final String measure;
		if (isTerminal) {
			measure = "terminal";
		} else {
			measure = "spot";
		}

		// Choose the simulation measure
		properties.put("measure", measure);

		//		if (dynamics == Dynamics.NORMAL) {
		//			properties.put("stateSpace", LIBORMarketModelFromCovarianceModel.StateSpace.NORMAL.name());
		//		} else {
		//			properties.put("stateSpace", LIBORMarketModelFromCovarianceModel.StateSpace.LOGNORMAL.name());
		//		}


		//		if (dynamics == Dynamics.NORMAL) {
		//			properties.put("stateSpace", "normal");
		//		} else {
		//			properties.put("stateSpace",  "lognormal");
		//		}

		/*
		 * Here we have to be careful: the fact that the STATE SPACE is "normal" means that you do not apply the exponential state-space
		 * transformation that you apply instead when you simulate log-normal processes, NOT that the dynamics of the process themselves
		 * are meant to be normal. Now, since in BlendedLocalVolatilityModel we simulate the model as (a L0 + (1-a)L) F, if a = 0
		 * (as it is the case for the log-normal DYNAMICS) and if F is specified by log-normal STATE SPACE, we would "multiply twice by L",
		 * and simulate something like	dL_t = sigma_L L_t^2 dW_t.
		 *	So, we have always to specify the STATE SPACE to be NORMAL when we use BlendedLocalVolatilityModel.
		 */
		properties.put("stateSpace", LIBORMarketModelFromCovarianceModel.StateSpace.NORMAL.name());

		/*
		 *  Empty array of calibration items and a RandomVariableFactory, to be given to the constructor of
		 *  LIBORMarketModelFromCovarianceModel
		 */
		final CalibrationProduct[] calibrationItems = new CalibrationProduct[0];
		final RandomVariableFactory randomVariableFactory = new RandomVariableFromArrayFactory();

		/*
		 *  LIBORMarketModelFromCovarianceModel is another class implementing LiborMarketModel, like LIBORMarketModelStandard.
		 *  It has the feature that you can specify some properties
		 */
		final ProcessModel LIBORMarketModel = new LIBORMarketModelFromCovarianceModel(
				LIBORPeriodDiscretization, null /* analyticModel */, forwardCurve, discountCurve, randomVariableFactory,
				covarianceModelBlended, calibrationItems, properties);
		//d\bar L = \bar L sigma dW
		//L=exp(\bar L)

		//dL=L^2 sigma dW_t

		//Step 9: create an Euler scheme of the LIBOR model defined above
		final BrownianMotion brownianMotion = new BrownianMotionFromMersenneRandomNumbers(
				timeDiscretization,
				LIBORPeriodDiscretization.getNumberOfTimes()-1,//no factor reduction for now
				numberOfPaths,
				seed // seed
				);

		final MonteCarloProcess process = new
				EulerSchemeFromProcessModel(LIBORMarketModel, brownianMotion);

		//Step 10: give the Euler scheme to the constructor of LIBORMonteCarloSimulationFromLIBORModel
		return new LIBORMonteCarloSimulationFromLIBORModel(process);
	}
}
