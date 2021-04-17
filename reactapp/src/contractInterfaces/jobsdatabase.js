import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x1e1e3214a312c3bb47cCeD4d244C23df02937Eae';

export default new web3.eth.Contract(jobs, address);