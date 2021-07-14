package OralExam;

import net.finmath.exception.CalculationException;
import net.finmath.montecarlo.interestrate.TermStructureMonteCarloSimulationModel;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.stochastic.RandomVariable;
import net.finmath.stochastic.Scalar;

public class LiborInArrears extends AbstractLIBORMonteCarloProduct {


	private final double	periodStartLibor; // T_i
	private final double	periodEndLibor; // T_{i+1}

	public LiborInArrears(double periodStartLibor, double periodEndLibor) {
		super();
		this.periodStartLibor = periodStartLibor;
		this.periodEndLibor = periodEndLibor;
	}

	/** This small implementation can take a Monte-Carlo Process Model (mostly Libor Monte-Carlo Process) and return the Value of a
	 * Floater in Arrears (mostly Libor in Arrears). It erbs the method getValue(TermStructureMonteCarloSimulationModel model).
	 *
	 */
	@Override
	public RandomVariable getValue(double evaluationTime, TermStructureMonteCarloSimulationModel model)
			throws CalculationException {

		// Get the value of the first LIBOR L_i at T_i: L(T_i,T_{i+1};T_i) -> we need only this libor!
		final RandomVariable	Libor = model.getLIBOR(periodStartLibor, periodStartLibor,
				periodEndLibor);

		final RandomVariable periodlength = new Scalar(periodEndLibor-periodStartLibor );

		// The floater pays Libor * Period length
		RandomVariable values = Libor.mult(periodlength);

		// Get numeraire at payment time: you then divide by N(T_{k+1})
		final RandomVariable	numeraire = model.getNumeraire(periodStartLibor); // We need the numeraire in payment date! This is T_i

		values = values.div(numeraire); // divide by numeraire in payment date by universal pricing theorem

		// Get numeraire at evaluation time: you multiply by N(0)
		final RandomVariable	numeraireAtEvaluationTime = model.getNumeraire(evaluationTime);

		values = values.mult(numeraireAtEvaluationTime);

		return values;
	}

}
