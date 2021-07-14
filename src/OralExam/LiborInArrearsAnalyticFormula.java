package OralExam;

public class LiborInArrearsAnalyticFormula {

	/**
	 * This method calculates and returns the value of a floater in Arrears
	 * model as an double assuming lognormal dynamics of the libor's
	 *
	 * @param initialForwardLibor,      	i.e. L_0 = L(T_i,T_i+1;0)
	 *
	 * @param liborVolatility,           	the volatility of the LIBOR process under
	 *                                  	the Black model(lognormal process)
	 *
	 * @param fixingPaymentDate,         	i.e. T_i (fixing=payment)
	 * @param endOfLiborPeriod,          	i.e. T_i+1
	 * @param endOfLibordiscountfactor		i.e P(T_i+1,0)
	 * @param fixingPaymentDiscountFactor	i.e. P(T_i;0)
	 * @param notional,                  	i.e. N
	 */
	public static double calculateLiborInArrearsFloaterAnalytic(double initialForwardLibor, double liborVolatility
			, double fixingPaymentDate , double endOfLiborPeriod, double endOfLibordiscountfactor , double fixingPaymentDiscountFactor) {

		final double periodLength = endOfLiborPeriod - fixingPaymentDate;  // fixingDate = paymentDate !

		final double firstPart = fixingPaymentDiscountFactor - endOfLibordiscountfactor ;
		final double convexityAdjustedPart = endOfLibordiscountfactor* periodLength *
				periodLength * initialForwardLibor * initialForwardLibor * Math.exp(fixingPaymentDate * liborVolatility);
		/*
		 * The value of a floater paying in advance ( Libor-in-Arrears )
		 * The analytical value is calculated by an convexity adjustment, where the first part is just the "normal" floater(paying in T_i+1)
		 * and the second part is resulting from the theorem in the lecture.
		 *
		 */
		return firstPart + convexityAdjustedPart ;
	}



}
