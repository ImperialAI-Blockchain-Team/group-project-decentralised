import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x1784f9C5b53888F07cFAeFEd8DD0C4ED4F2E60FF';

export default new web3.eth.Contract(jobs, address);
