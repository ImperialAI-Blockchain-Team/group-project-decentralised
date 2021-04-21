import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0xE84d2ebCFD02686066D9671f034ca93953099057';

export default new web3.eth.Contract(jobs, address);
